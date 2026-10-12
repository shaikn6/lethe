package memory

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/shaikn6/lethe/internal/policy"
)

var errDisk = errors.New("disk error")

// flakyFile fails writes on demand, writing half the bytes first the way a
// full or failing disk does.
type flakyFile struct {
	journalFile
	failWrites   bool
	failTruncate bool
}

func (f *flakyFile) Write(p []byte) (int, error) {
	if f.failWrites {
		n, _ := f.journalFile.Write(p[:len(p)/2])
		return n, errDisk
	}
	return f.journalFile.Write(p)
}

func (f *flakyFile) Truncate(size int64) error {
	if f.failTruncate {
		return errDisk
	}
	return f.journalFile.Truncate(size)
}

func breakDisk(s *Store) *flakyFile {
	flaky := &flakyFile{journalFile: s.journal.file, failWrites: true}
	s.journal.file = flaky
	return flaky
}

func TestFailedWriteIsRolledBackAndTheJournalStaysReadable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.journal")
	s := open(t, Options{JournalPath: path})
	remember(t, s, Memory{Text: "written before the disk problem"})

	disk := breakDisk(s)
	if _, err := s.Remember(Memory{Namespace: "ns", Text: "lost to the disk problem"}); !errors.Is(err, errDisk) {
		t.Fatalf("Remember on a failing disk = %v, want the disk error", err)
	}
	if got := recall(t, s, Query{Text: "lost disk problem"}); len(got.Memories) != 1 {
		t.Fatalf("a memory that was never written is being served: %v", texts(got))
	}
	disk.failWrites = false
	remember(t, s, Memory{Text: "written after the disk recovered"})
	if st := s.Stats(); st.Memories != 2 || st.JournalErrors != 1 || !st.JournalHealthy {
		t.Fatalf("stats = %+v", st)
	}
	s.Close()

	reopened := open(t, Options{JournalPath: path})
	if st := reopened.Stats(); st.Memories != 2 {
		t.Fatalf("after restart: %d memories, want 2 (journal must not hold a half-written line)", st.Memories)
	}
}

func TestJournalThatCannotBeRolledBackRefusesFurtherWrites(t *testing.T) {
	s := open(t, Options{JournalPath: filepath.Join(t.TempDir(), "store.journal")})
	kept := remember(t, s, Memory{Text: "safe memory"})
	disk := breakDisk(s)
	disk.failTruncate = true
	if _, err := s.Remember(Memory{Namespace: "ns", Text: "x"}); err == nil {
		t.Fatal("Remember succeeded on a failing disk")
	}
	disk.failWrites, disk.failTruncate = false, false
	if _, err := s.Remember(Memory{Namespace: "ns", Text: "y"}); err == nil || !strings.Contains(err.Error(), "unusable") {
		t.Fatalf("Remember after an unrecoverable journal error = %v, want it refused", err)
	}
	if err := s.Forget("ns", kept.ID); err == nil {
		t.Fatal("Forget succeeded with an unusable journal")
	}
	if st := s.Stats(); st.JournalHealthy || st.Memories != 1 {
		t.Fatalf("stats = %+v, want unhealthy journal and the one safe memory", st)
	}
	if got := recall(t, s, Query{Text: "safe memory"}); len(got.Memories) != 1 {
		t.Fatalf("reads must keep working: %v", texts(got))
	}
}

func TestFeedbackAndWeightsNeverRunAheadOfTheJournal(t *testing.T) {
	s := open(t, Options{JournalPath: filepath.Join(t.TempDir(), "store.journal")})
	for i := range 5 {
		remember(t, s, Memory{Text: fmt.Sprintf("note %d about the topic", i), Embedding: []float32{float32(i), 1}})
	}
	r := recall(t, s, Query{Text: "topic note", Embedding: []float32{1, 1}, Explore: true})
	before := s.Stats()

	disk := breakDisk(s)
	if err := s.Feedback(r.ID, 1); !errors.Is(err, errDisk) {
		t.Fatalf("Feedback on a failing disk = %v, want the disk error", err)
	}
	var other policy.Features
	other[policy.Recency] = 3
	if err := s.SetWeights(other); !errors.Is(err, errDisk) {
		t.Fatalf("SetWeights on a failing disk = %v, want the disk error", err)
	}
	after := s.Stats()
	if after.Weights != before.Weights || after.Updates != before.Updates || after.Baseline != before.Baseline || after.Pending != 1 {
		t.Fatalf("state changed although nothing was journaled:\n got %+v\nwant %+v", after, before)
	}

	disk.failWrites = false
	if err := s.Feedback(r.ID, 1); err != nil {
		t.Fatalf("retry after the disk recovered = %v", err)
	}
	if st := s.Stats(); st.Updates != before.Updates+1 || st.Pending != 0 {
		t.Fatalf("stats after retry = %+v", st)
	}
}

func TestForgottenTextIsErasedFromMemoryAndDisk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.journal")
	s := open(t, Options{JournalPath: path})
	remember(t, s, Memory{Text: "the gate code is on the notice board"})
	secret := remember(t, s, Memory{Text: "card number zebra4417 expires soon", Metadata: map[string]string{"pin": "quokka9921"}})
	onDisk := func() string {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	if !strings.Contains(onDisk(), "zebra4417") {
		t.Fatal("test setup: the secret should be in the journal before it is forgotten")
	}
	if err := s.Forget("ns", secret.ID); err != nil {
		t.Fatal(err)
	}
	sp := s.spaces["ns"]
	for _, e := range sp.entries {
		if strings.Contains(e.Text, "zebra4417") || e.Metadata["pin"] != "" {
			t.Fatalf("forgotten memory is still in RAM: %+v", e.Memory)
		}
	}
	if got := sp.ix.Search("zebra4417", 5); len(got) != 0 {
		t.Fatal("forgotten word is still searchable")
	}
	if err := s.Compact(); err != nil {
		t.Fatal(err)
	}
	for _, leaked := range []string{"zebra4417", "quokka9921", secret.ID} {
		if strings.Contains(onDisk(), leaked) {
			t.Fatalf("%q is still in the journal after compaction", leaked)
		}
	}
	if !strings.Contains(onDisk(), "gate code") {
		t.Fatal("compaction dropped a live memory")
	}
}

func TestForgottenTextDoesNotSurviveARestartOrManyForgets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.journal")
	s := open(t, Options{JournalPath: path, CompactAfter: 3})
	var ids []string
	for i := range 3 {
		ids = append(ids, remember(t, s, Memory{Text: fmt.Sprintf("disposable wombat%d note", i)}).ID)
	}
	keep := remember(t, s, Memory{Text: "permanent note"})
	for _, id := range ids {
		if err := s.Forget("ns", id); err != nil {
			t.Fatal(err)
		}
	}
	data, _ := os.ReadFile(path)
	if bytes.Contains(data, []byte("wombat")) {
		t.Fatal("journal still holds forgotten text after CompactAfter forgets")
	}

	late := remember(t, s, Memory{Text: "another disposable numbat note"})
	if err := s.Forget("ns", late.ID); err != nil {
		t.Fatal(err)
	}
	s.Close()
	data, _ = os.ReadFile(path)
	if bytes.Contains(data, []byte("numbat")) {
		t.Fatal("journal still holds forgotten text after Close")
	}
	reopened := open(t, Options{JournalPath: path})
	got := recall(t, reopened, Query{Text: "permanent note"})
	if len(got.Memories) != 1 || got.Memories[0].ID != keep.ID {
		t.Fatalf("live memory lost across compaction and restart: %v", texts(got))
	}
}

func TestCompactionKeepsRankingAndLearningAndShrinksTheJournal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.journal")
	s := open(t, Options{JournalPath: path})
	for i := range 40 {
		remember(t, s, Memory{Text: fmt.Sprintf("meeting note %d about the budget", i), Embedding: []float32{float32(i % 7), float32(i % 5), 1}, CreatedAt: day0.AddDate(0, 0, i)})
	}
	for i := range 300 {
		r := recall(t, s, Query{Text: "latest budget meeting note", Embedding: []float32{1, 2, 1}, Explore: true})
		if err := s.Feedback(r.ID, float64(i%2)); err != nil {
			t.Fatal(err)
		}
	}
	query := Query{Text: "latest budget meeting note", Embedding: []float32{1, 2, 1}, K: 10}
	before, stats := recall(t, s, query), s.Stats()
	sizeBefore := fileSize(t, path)
	if err := s.Compact(); err != nil {
		t.Fatal(err)
	}
	if sizeAfter := fileSize(t, path); sizeAfter*3 > sizeBefore {
		t.Fatalf("journal went from %d to %d bytes; expected it to shrink by well over half", sizeBefore, sizeAfter)
	}
	sameRecall(t, "after compaction", recall(t, s, query), before)
	remember(t, s, Memory{Text: "written after the compaction"})
	s.Close()

	reopened := open(t, Options{JournalPath: path})
	sameRecall(t, "after compaction and restart", recall(t, reopened, query), before)
	if got := reopened.Stats(); got.Memories != 41 || got.Weights != stats.Weights || got.Updates != stats.Updates {
		t.Fatalf("stats after restart = %+v, want 41 memories and the learned state %+v", got, stats)
	}
}

func TestStoreRefusesNewMemoriesWhenFull(t *testing.T) {
	s := open(t, Options{MaxMemories: 2})
	first := remember(t, s, Memory{Text: "one"})
	remember(t, s, Memory{Namespace: "other", Text: "two"})
	if _, err := s.Remember(Memory{Namespace: "ns", Text: "three"}); !errors.Is(err, ErrFull) {
		t.Fatalf("Remember beyond the limit = %v, want ErrFull", err)
	}
	if err := s.Forget("ns", first.ID); err != nil {
		t.Fatal(err)
	}
	remember(t, s, Memory{Text: "three fits after one was forgotten"})
}

func TestRejectsOversizedMetadata(t *testing.T) {
	s := open(t, Options{})
	for name, metadata := range map[string]map[string]string{
		"long key":   {strings.Repeat("k", MaxMetadataKeyLen+1): "v"},
		"long value": {"k": strings.Repeat("v", MaxMetadataValLen+1)},
	} {
		if _, err := s.Remember(Memory{Namespace: "ns", Text: "x", Metadata: metadata}); !errors.Is(err, ErrInvalid) {
			t.Errorf("%s: %v, want ErrInvalid", name, err)
		}
	}
}

func TestCompactionDuringConcurrentUse(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.journal")
	s := open(t, Options{JournalPath: path, CompactAfter: 20})
	const workers, rounds = 8, 120
	var wg sync.WaitGroup
	var mu sync.Mutex
	kept := 0
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
				if err := s.Compact(); err != nil {
					t.Error(err)
				}
			}
		}
	}()
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ns := fmt.Sprintf("ns%d", w%3)
			for i := range rounds {
				m, err := s.Remember(Memory{Namespace: ns, Text: fmt.Sprintf("worker %d item %d on the shared subject", w, i), Embedding: []float32{float32(w), float32(i), 1}})
				if err != nil {
					t.Error(err)
					return
				}
				r, err := s.Recall(Query{Namespace: ns, Text: "shared subject item", Embedding: []float32{1, 1, 1}, K: 5, Explore: true})
				if err != nil {
					t.Error(err)
					return
				}
				for _, got := range r.Memories {
					if got.Text == "" {
						t.Error("recall returned an erased memory")
					}
				}
				if err := s.Feedback(r.ID, float64(i%2)); err != nil {
					t.Error(err)
					return
				}
				if i%3 == 0 {
					if err := s.Forget(ns, m.ID); err != nil {
						t.Error(err)
						return
					}
				} else {
					mu.Lock()
					kept++
					mu.Unlock()
				}
			}
		}()
	}
	wg.Wait()
	close(stop)
	st := s.Stats()
	if st.Memories != kept || st.Updates != workers*rounds || st.JournalErrors != 0 {
		t.Fatalf("stats = %+v, want %d memories and %d updates", st, kept, workers*rounds)
	}
	s.Close()
	reopened := open(t, Options{JournalPath: path})
	if got := reopened.Stats(); got.Memories != kept || got.Updates != st.Updates || got.Weights != st.Weights {
		t.Fatalf("after restart: %+v\nwant %d memories, state %+v", got, kept, st)
	}
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.Size()
}

func sameRecall(t *testing.T, when string, got, want Recall) {
	t.Helper()
	if len(got.Memories) != len(want.Memories) {
		t.Fatalf("%s: %d memories, want %d", when, len(got.Memories), len(want.Memories))
	}
	for i := range got.Memories {
		if got.Memories[i].ID != want.Memories[i].ID || got.Memories[i].Score != want.Memories[i].Score {
			t.Fatalf("%s: position %d is %+v, want %+v", when, i, got.Memories[i], want.Memories[i])
		}
	}
}
