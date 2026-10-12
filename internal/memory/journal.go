package memory

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/shaikn6/lethe/internal/policy"
)

// record is one line of the journal. Exactly one of the pointer fields is set.
type record struct {
	Remember *Memory      `json:"remember,omitempty"`
	Forget   *forgetEntry `json:"forget,omitempty"`
	Learn    *learnEntry  `json:"learn,omitempty"`

	// Set next to Remember in a compacted journal: what the memory had
	// earned from feedback before the compaction.
	Recalls   int     `json:"recalls,omitempty"`
	RewardSum float64 `json:"reward_sum,omitempty"`
}

type forgetEntry struct {
	Namespace string `json:"namespace"`
	ID        string `json:"id"`
}

type learnEntry struct {
	Weights   policy.Features `json:"weights"`
	Baseline  float64         `json:"baseline"`
	Updates   int             `json:"updates"`
	Namespace string          `json:"namespace,omitempty"`
	IDs       []string        `json:"ids,omitempty"`
	Reward    float64         `json:"reward,omitempty"`
}

// journalFile is the part of *os.File the journal uses, so tests can make
// writes fail.
type journalFile interface {
	io.Writer
	Sync() error
	Truncate(size int64) error
	Seek(offset int64, whence int) (int64, error)
	Close() error
}

// journal is an append-only log of every change. Replaying it rebuilds the
// store, so a process can be killed at any point and lose at most the write
// that was in flight.
type journal struct {
	mu     sync.Mutex
	file   journalFile
	path   string
	sync   bool
	size   int64 // bytes of complete records
	broken error // set once the file can no longer be trusted
}

// openJournal replays the journal at path through apply and leaves it open
// for appending. A final line cut short by a crash is dropped; damage
// anywhere else is reported rather than skipped, because skipping it would
// silently lose memories.
func openJournal(path string, fsync bool, apply func(record) error) (*journal, error) {
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}
	reader := bufio.NewReaderSize(file, 1<<20)
	var good int64
	for line := 1; ; line++ {
		data, err := reader.ReadBytes('\n')
		if errors.Is(err, io.EOF) {
			break // anything left in data is a torn final write
		}
		if err != nil {
			file.Close()
			return nil, err
		}
		var rec record
		if len(bytes.TrimSpace(data)) > 0 {
			if err := json.Unmarshal(data, &rec); err != nil {
				file.Close()
				return nil, fmt.Errorf("journal %s is damaged at line %d: %w", path, line, err)
			}
			if err := apply(rec); err != nil {
				file.Close()
				return nil, fmt.Errorf("journal %s line %d: %w", path, line, err)
			}
		}
		good += int64(len(data))
	}
	if err := file.Truncate(good); err != nil {
		file.Close()
		return nil, err
	}
	if _, err := file.Seek(good, io.SeekStart); err != nil {
		file.Close()
		return nil, err
	}
	return &journal{file: file, path: path, sync: fsync, size: good}, nil
}

// append writes one record. If the write fails part-way the file is rolled
// back to the last complete record, so a half-written line can never end up
// in the middle of the journal. If even the rollback fails, the journal is
// marked broken and refuses further writes rather than risk corrupting it.
func (j *journal) append(rec record) error {
	data, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.broken != nil {
		return j.broken
	}
	n, err := j.file.Write(data)
	if err == nil && j.sync {
		err = j.file.Sync()
	}
	if err != nil {
		if rollback := j.rewind(); rollback != nil {
			j.broken = fmt.Errorf("journal is unusable: write failed (%v) and rollback failed (%v)", err, rollback)
		}
		return err
	}
	j.size += int64(n)
	return nil
}

func (j *journal) rewind() error {
	if err := j.file.Truncate(j.size); err != nil {
		return err
	}
	_, err := j.file.Seek(j.size, io.SeekStart)
	return err
}

// replace swaps the journal for a new one holding exactly the records that
// write emits. The new file is complete and on disk before it takes the old
// one's place, so a crash leaves either the old journal or the new one.
func (j *journal) replace(write func(emit func(record) error) error) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.broken != nil {
		return j.broken
	}
	tmpPath := j.path + ".compact"
	tmp, err := os.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	discard := func(err error) error {
		tmp.Close()
		os.Remove(tmpPath)
		return err
	}
	out := bufio.NewWriterSize(tmp, 1<<20)
	var size int64
	emit := func(rec record) error {
		data, err := json.Marshal(rec)
		if err != nil {
			return err
		}
		n, err := out.Write(append(data, '\n'))
		size += int64(n)
		return err
	}
	if err := write(emit); err != nil {
		return discard(err)
	}
	if err := out.Flush(); err != nil {
		return discard(err)
	}
	if err := tmp.Sync(); err != nil {
		return discard(err)
	}
	if err := os.Rename(tmpPath, j.path); err != nil {
		return discard(err)
	}
	if dir, err := os.Open(filepath.Dir(j.path)); err == nil {
		_ = dir.Sync()
		dir.Close()
	}
	j.file.Close()
	j.file, j.size = tmp, size
	return nil
}

func (j *journal) healthy() bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.broken == nil
}

func (j *journal) close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.file.Close()
}
