package memory

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/shaikn6/lethe/internal/policy"
)

var day0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func open(t *testing.T, opts Options) *Store {
	t.Helper()
	if opts.Seed == 0 {
		opts.Seed = 42
	}
	if opts.Now == nil {
		opts.Now = func() time.Time { return day0.AddDate(0, 0, 60) }
	}
	s, err := Open(opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func remember(t *testing.T, s *Store, m Memory) Memory {
	t.Helper()
	if m.Namespace == "" {
		m.Namespace = "ns"
	}
	out, err := s.Remember(m)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func recall(t *testing.T, s *Store, q Query) Recall {
	t.Helper()
	if q.Namespace == "" {
		q.Namespace = "ns"
	}
	if q.K == 0 {
		q.K = 3
	}
	out, err := s.Recall(q)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func texts(r Recall) []string {
	out := make([]string, len(r.Memories))
	for i, m := range r.Memories {
		out[i] = m.Text
	}
	return out
}

func TestRecallByKeyword(t *testing.T) {
	s := open(t, Options{})
	remember(t, s, Memory{Text: "Priya is allergic to peanuts"})
	remember(t, s, Memory{Text: "The quarterly report is due on Friday"})
	got := recall(t, s, Query{Text: "what is Priya allergic to", K: 1})
	if len(got.Memories) != 1 || got.Memories[0].Text != "Priya is allergic to peanuts" {
		t.Fatalf("recall = %v", texts(got))
	}
	if got.ID == "" || got.Memories[0].ID == "" || got.Memories[0].CreatedAt.IsZero() {
		t.Fatalf("missing ids or timestamp: %+v", got)
	}
}

func TestRecallBySemanticsWithoutSharedWords(t *testing.T) {
	s := open(t, Options{})
	remember(t, s, Memory{Text: "She cannot eat groundnuts", Embedding: []float32{1, 0, 0}})
	remember(t, s, Memory{Text: "The server room is on floor two", Embedding: []float32{0, 1, 0}})
	got := recall(t, s, Query{Text: "peanut allergy", Embedding: []float32{0.9, 0.1, 0}, K: 1})
	if len(got.Memories) != 1 || got.Memories[0].Text != "She cannot eat groundnuts" {
		t.Fatalf("recall = %v", texts(got))
	}
}

func TestKeywordSearchStillWorksWhenQueryHasNoEmbedding(t *testing.T) {
	s := open(t, Options{})
	remember(t, s, Memory{Text: "Backup runs at midnight", Embedding: []float32{1, 0}})
	got := recall(t, s, Query{Text: "when does the backup run"})
	if len(got.Memories) != 1 {
		t.Fatalf("recall = %v, want the backup memory", texts(got))
	}
}

func TestNamespacesAreIsolated(t *testing.T) {
	s := open(t, Options{})
	remember(t, s, Memory{Namespace: "alice", Text: "door code is 4411"})
	if got := recall(t, s, Query{Namespace: "bob", Text: "door code"}); len(got.Memories) != 0 {
		t.Fatalf("bob can read alice's memory: %v", texts(got))
	}
}

func TestForget(t *testing.T) {
	s := open(t, Options{})
	m := remember(t, s, Memory{Text: "temporary note about the launch"})
	if err := s.Forget("ns", m.ID); err != nil {
		t.Fatal(err)
	}
	if got := recall(t, s, Query{Text: "launch note"}); len(got.Memories) != 0 {
		t.Fatalf("forgotten memory recalled: %v", texts(got))
	}
	if err := s.Forget("ns", m.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("second Forget = %v, want ErrNotFound", err)
	}
	if err := s.Forget("other", "x"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Forget in unknown namespace = %v, want ErrNotFound", err)
	}
}

func TestRejectsInvalidInput(t *testing.T) {
	s := open(t, Options{})
	remember(t, s, Memory{Text: "seed", Embedding: []float32{1, 0}})
	nan := float32(0)
	nan /= nan
	bad := map[string]Memory{
		"no namespace":       {Text: "x"},
		"empty text":         {Namespace: "ns"},
		"text too long":      {Namespace: "ns", Text: strings.Repeat("a", MaxTextBytes+1)},
		"importance too big": {Namespace: "ns", Text: "x", Importance: 2},
		"wrong dimension":    {Namespace: "ns", Text: "x", Embedding: []float32{1, 0, 0}},
		"nan in embedding":   {Namespace: "ns", Text: "x", Embedding: []float32{nan, 0}},
		"embedding too wide": {Namespace: "ns", Text: "x", Embedding: make([]float32, MaxEmbeddingDim+1)},
	}
	for name, m := range bad {
		if _, err := s.Remember(m); !errors.Is(err, ErrInvalid) {
			t.Errorf("Remember(%s) = %v, want ErrInvalid", name, err)
		}
	}
	for name, q := range map[string]Query{
		"k zero":       {Namespace: "ns", Text: "x"},
		"k too large":  {Namespace: "ns", Text: "x", K: MaxK + 1},
		"no namespace": {Text: "x", K: 1},
	} {
		if _, err := s.Recall(q); !errors.Is(err, ErrInvalid) {
			t.Errorf("Recall(%s) = %v, want ErrInvalid", name, err)
		}
	}
	if st := s.Stats(); st.Memories != 1 {
		t.Fatalf("invalid input changed the store: %+v", st)
	}
}

func TestFeedbackIsAcceptedOncePerRecall(t *testing.T) {
	s := open(t, Options{})
	remember(t, s, Memory{Text: "alpha"})
	r := recall(t, s, Query{Text: "alpha", Explore: true})
	if err := s.Feedback(r.ID, 1.5, nil); !errors.Is(err, ErrInvalid) {
		t.Fatalf("out-of-range reward = %v, want ErrInvalid", err)
	}
	if err := s.Feedback(r.ID, 1, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.Feedback(r.ID, 1, nil); !errors.Is(err, ErrNotFound) {
		t.Fatalf("second feedback = %v, want ErrNotFound", err)
	}
	if err := s.Feedback("nope", 1, nil); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown recall = %v, want ErrNotFound", err)
	}
	empty := recall(t, s, Query{Text: "nothing matches this", Explore: true})
	if err := s.Feedback(empty.ID, 0, nil); err != nil {
		t.Fatalf("feedback on an empty recall = %v, want nil", err)
	}
}

// Thirty status updates share the same words; only the newest answers "what
// is the latest status". Nothing tells the store that except reward.
func TestLearnsFromFeedbackWhichMemoryToReturn(t *testing.T) {
	s := open(t, Options{})
	var newest string
	for i := range 30 {
		m := remember(t, s, Memory{
			Text:      fmt.Sprintf("project status update: milestone %d is done", i),
			CreatedAt: day0.AddDate(0, 0, i),
		})
		newest = m.ID
	}
	q := Query{Text: "what is the latest project status", AsOf: day0.AddDate(0, 0, 29)}
	top := func() bool {
		r := recall(t, s, q)
		return len(r.Memories) > 0 && r.Memories[0].ID == newest
	}
	if top() {
		t.Fatal("untrained store already returns the newest update first; the test proves nothing")
	}
	q.Explore = true
	for range 600 {
		r := recall(t, s, q)
		reward := 0.0
		for _, m := range r.Memories {
			if m.ID == newest {
				reward = 1
			}
		}
		if err := s.Feedback(r.ID, reward, nil); err != nil {
			t.Fatal(err)
		}
	}
	q.Explore = false
	if !top() {
		t.Fatalf("after training the newest update is not first; weights %v", s.Stats().Weights)
	}
}

func TestJournalRestoresMemoriesAndLearning(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.journal")
	s := open(t, Options{JournalPath: path})
	keep := remember(t, s, Memory{Text: "the wifi password is on the fridge", Embedding: []float32{1, 0}, Metadata: map[string]string{"source": "chat"}})
	gone := remember(t, s, Memory{Text: "old wifi password was hunter2", Embedding: []float32{0, 1}})
	remember(t, s, Memory{Text: "router is in the hallway closet", Embedding: []float32{1, 1}})
	if err := s.Forget("ns", gone.ID); err != nil {
		t.Fatal(err)
	}
	for range 25 {
		r := recall(t, s, Query{Text: "wifi password", Embedding: []float32{1, 0.1}, Explore: true})
		if err := s.Feedback(r.ID, 1, nil); err != nil {
			t.Fatal(err)
		}
	}
	query := Query{Text: "wifi password router", Embedding: []float32{1, 0.2}}
	before, stats := recall(t, s, query), s.Stats()
	s.Close()

	reopened := open(t, Options{JournalPath: path})
	after, restored := recall(t, reopened, query), reopened.Stats()
	if restored.Memories != 2 || restored.Updates != stats.Updates || restored.Weights != stats.Weights || restored.Baseline != stats.Baseline {
		t.Fatalf("stats after restart:\n got %+v\nwant %+v", restored, stats)
	}
	if len(after.Memories) != len(before.Memories) {
		t.Fatalf("recall changed after restart: %v vs %v", texts(after), texts(before))
	}
	for i := range after.Memories {
		a, b := after.Memories[i], before.Memories[i]
		if a.ID != b.ID || a.Score != b.Score || a.ID == gone.ID {
			t.Fatalf("recall changed after restart:\n got %+v\nwant %+v", after.Memories, before.Memories)
		}
	}
	if after.Memories[0].ID != keep.ID || after.Memories[0].Metadata["source"] != "chat" {
		t.Fatalf("top memory after restart = %+v", after.Memories[0])
	}
}

func TestJournalSurvivesAWriteCutShortByACrash(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.journal")
	s := open(t, Options{JournalPath: path})
	remember(t, s, Memory{Text: "first complete memory"})
	remember(t, s, Memory{Text: "second complete memory"})
	s.Close()

	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	file.WriteString(`{"remember":{"id":"abc","namespace":"ns","text":"half writ`)
	file.Close()

	reopened := open(t, Options{JournalPath: path})
	if st := reopened.Stats(); st.Memories != 2 {
		t.Fatalf("memories after torn write = %d, want 2", st.Memories)
	}
	remember(t, reopened, Memory{Text: "third memory written after recovery"})
	reopened.Close()

	again := open(t, Options{JournalPath: path})
	if st := again.Stats(); st.Memories != 3 {
		t.Fatalf("memories after second restart = %d, want 3", st.Memories)
	}
}

func TestJournalDamageInTheMiddleIsReportedNotSkipped(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.journal")
	s := open(t, Options{JournalPath: path})
	remember(t, s, Memory{Text: "one"})
	remember(t, s, Memory{Text: "two"})
	s.Close()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	damaged := append([]byte("{not json}\n"), data...)
	if err := os.WriteFile(path, damaged, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(Options{JournalPath: path}); err == nil || !strings.Contains(err.Error(), "damaged at line 1") {
		t.Fatalf("Open = %v, want a damage report for line 1", err)
	}
}

func TestPendingRecallsAreBounded(t *testing.T) {
	s := open(t, Options{MaxPending: 10})
	remember(t, s, Memory{Text: "alpha"})
	first := recall(t, s, Query{Text: "alpha"})
	for range 50 {
		recall(t, s, Query{Text: "alpha"})
	}
	if st := s.Stats(); st.Pending != 10 {
		t.Fatalf("pending = %d, want 10", st.Pending)
	}
	if err := s.Feedback(first.ID, 1, nil); !errors.Is(err, ErrNotFound) {
		t.Fatalf("feedback for an evicted recall = %v, want ErrNotFound", err)
	}
}

func TestConcurrentUse(t *testing.T) {
	s := open(t, Options{JournalPath: filepath.Join(t.TempDir(), "store.journal")})
	const workers, rounds = 16, 150
	var wg sync.WaitGroup
	var mu sync.Mutex
	kept := 0
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ns := fmt.Sprintf("ns%d", w%4)
			for i := range rounds {
				m, err := s.Remember(Memory{Namespace: ns, Text: fmt.Sprintf("worker %d note %d about the shared topic", w, i), Embedding: []float32{float32(w), float32(i), 1}})
				if err != nil {
					t.Error(err)
					return
				}
				r, err := s.Recall(Query{Namespace: ns, Text: "shared topic note", Embedding: []float32{1, 1, 1}, K: 5, Explore: i%2 == 0})
				if err != nil {
					t.Error(err)
					return
				}
				if err := s.Feedback(r.ID, float64(i%3)/2, nil); err != nil {
					t.Error(err)
					return
				}
				if i%5 == 0 {
					if err := s.Forget(ns, m.ID); err != nil {
						t.Error(err)
						return
					}
				} else {
					mu.Lock()
					kept++
					mu.Unlock()
				}
				s.Stats()
			}
		}()
	}
	wg.Wait()
	st := s.Stats()
	if st.Memories != kept || st.Namespaces != 4 || st.Pending != 0 || st.JournalErrors != 0 {
		t.Fatalf("stats = %+v, want %d memories in 4 namespaces, nothing pending", st, kept)
	}
	path := s.opts.JournalPath
	s.Close()
	reopened := open(t, Options{JournalPath: path})
	if got := reopened.Stats(); got.Memories != kept || got.Updates != st.Updates {
		t.Fatalf("after restart: %+v, want %d memories and %d updates", got, kept, st.Updates)
	}
}

func TestSetWeightsChangesRankingAndSurvivesRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.journal")
	s := open(t, Options{JournalPath: path})
	old := remember(t, s, Memory{Text: "status report", CreatedAt: day0})
	recent := remember(t, s, Memory{Text: "status report", CreatedAt: day0.AddDate(0, 0, 59)})
	q := Query{Text: "status report", K: 1}
	if got := recall(t, s, q); got.Memories[0].ID != old.ID {
		t.Fatalf("default ranking should tie-break to the older memory, got %+v", got.Memories[0])
	}

	var recencyOnly policy.Features
	recencyOnly[policy.Recency] = 5
	if err := s.SetWeights(recencyOnly); err != nil {
		t.Fatal(err)
	}
	if got := recall(t, s, q); got.Memories[0].ID != recent.ID {
		t.Fatalf("recency ranking returned %+v", got.Memories[0])
	}

	bad := recencyOnly
	bad[policy.Keyword] = math.NaN()
	if err := s.SetWeights(bad); !errors.Is(err, ErrInvalid) {
		t.Fatalf("SetWeights(NaN) = %v, want ErrInvalid", err)
	}
	bad[policy.Keyword] = policy.MaxWeight * 2
	if err := s.SetWeights(bad); !errors.Is(err, ErrInvalid) {
		t.Fatalf("SetWeights(too large) = %v, want ErrInvalid", err)
	}

	s.Close()
	reopened := open(t, Options{JournalPath: path})
	if got := reopened.Stats().Weights; got != recencyOnly {
		t.Fatalf("weights after restart = %v, want %v", got, recencyOnly)
	}
}

func TestFeedbackCanNameTheMemoriesThatHelped(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.journal")
	s := open(t, Options{JournalPath: path})
	for i := range 6 {
		remember(t, s, Memory{Text: fmt.Sprintf("holiday plan option %d", i), Embedding: []float32{float32(i), 1}})
	}
	q := Query{Text: "holiday plan option", Embedding: []float32{2, 1}, K: 3, Explore: true}
	r := recall(t, s, q)
	if err := s.Feedback(r.ID, 1, []string{"not-in-this-recall"}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("feedback naming a memory that was not returned = %v, want ErrInvalid", err)
	}
	helped := r.Memories[1].ID
	if err := s.Feedback(r.ID, 1, []string{helped}); err != nil {
		t.Fatal(err)
	}
	utility := func(st *Store, id string) float64 {
		sp := st.spaces["ns"]
		return sp.entries[sp.byID[id]].utility()
	}
	if got, other := utility(s, helped), utility(s, r.Memories[0].ID); got <= 0.5 || other >= 0.5 {
		t.Fatalf("utility of the helpful memory = %.2f, of an unhelpful one = %.2f; want above and below 0.5", got, other)
	}
	want, stats := utility(s, helped), s.Stats()
	s.Close()
	reopened := open(t, Options{JournalPath: path})
	if got := utility(reopened, helped); got != want || reopened.Stats().Weights != stats.Weights {
		t.Fatalf("per-memory credit not restored: utility %.3f want %.3f", got, want)
	}
}

func TestEventsDescribeEveryOperationWithoutContent(t *testing.T) {
	var events []Event
	s := open(t, Options{OnEvent: func(e Event) { events = append(events, e) }})
	m := remember(t, s, Memory{Text: "private detail platypus31"})
	r := recall(t, s, Query{Text: "private detail", Explore: true})
	if err := s.Feedback(r.ID, 1, []string{m.ID}); err != nil {
		t.Fatal(err)
	}
	if err := s.Forget("ns", m.ID); err != nil {
		t.Fatal(err)
	}
	s.Recall(Query{Namespace: "", Text: "invalid"}) // failures emit nothing

	var types []string
	for _, e := range events {
		types = append(types, e.Type)
		if e.At.IsZero() || e.Namespace != "ns" {
			t.Fatalf("event missing time or namespace: %+v", e)
		}
		if raw := fmt.Sprintf("%+v", e); strings.Contains(raw, "platypus31") || strings.Contains(raw, "private detail") {
			t.Fatalf("event leaks content: %s", raw)
		}
	}
	if strings.Join(types, ",") != "remember,recall,feedback,forget" {
		t.Fatalf("event types = %v", types)
	}
	if fb := events[2]; fb.RecallID != r.ID || fb.Reward == nil || *fb.Reward != 1 || fb.Useful[0] != m.ID {
		t.Fatalf("feedback event = %+v", fb)
	}
}
