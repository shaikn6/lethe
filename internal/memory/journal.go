package memory

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/shaikn6/lethe/internal/policy"
)

// record is one line of the journal. Exactly one of the pointer fields is set.
type record struct {
	Remember *Memory      `json:"remember,omitempty"`
	Forget   *forgetEntry `json:"forget,omitempty"`
	Learn    *learnEntry  `json:"learn,omitempty"`
}

type forgetEntry struct {
	Namespace string `json:"namespace"`
	ID        string `json:"id"`
}

type learnEntry struct {
	Weights   policy.Features `json:"weights"`
	Baseline  float64         `json:"baseline"`
	Updates   int             `json:"updates"`
	Namespace string          `json:"namespace"`
	IDs       []string        `json:"ids"`
	Reward    float64         `json:"reward"`
}

// journal is an append-only log of every change. Replaying it rebuilds the
// store, so a process can be killed at any point and lose at most the write
// that was in flight.
type journal struct {
	mu   sync.Mutex
	file *os.File
	sync bool
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
	return &journal{file: file, sync: fsync}, nil
}

func (j *journal) append(rec record) error {
	data, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	j.mu.Lock()
	defer j.mu.Unlock()
	if _, err := j.file.Write(data); err != nil {
		return err
	}
	if j.sync {
		return j.file.Sync()
	}
	return nil
}

func (j *journal) close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.file.Close()
}
