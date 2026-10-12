package events

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/shaikn6/lethe/internal/memory"
)

type fakeProducer struct {
	mu      sync.Mutex
	records [][2]string
	fail    bool
	block   chan struct{} // when set, Produce waits on it
	closed  bool
}

func (f *fakeProducer) Produce(key, value []byte, done func(error)) {
	if f.block != nil {
		<-f.block
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.fail {
		done(errors.New("broker unavailable"))
		return
	}
	f.records = append(f.records, [2]string{string(key), string(value)})
	done(nil)
}

func (f *fakeProducer) Flush(context.Context) error { return nil }
func (f *fakeProducer) Close()                      { f.closed = true }

func TestEventsReachTheProducerKeyedByNamespace(t *testing.T) {
	producer := &fakeProducer{}
	stream := NewStream(producer, 16)
	reward := 1.0
	stream.Publish(memory.Event{Type: "recall", Namespace: "alice", RecallID: "r1", MemoryIDs: []string{"m1", "m2"}})
	stream.Publish(memory.Event{Type: "feedback", Namespace: "alice", RecallID: "r1", Reward: &reward, Useful: []string{"m2"}})
	if err := stream.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(producer.records) != 2 || producer.records[0][0] != "alice" || !producer.closed {
		t.Fatalf("records = %v, closed = %v", producer.records, producer.closed)
	}
	var got memory.Event
	if err := json.Unmarshal([]byte(producer.records[1][1]), &got); err != nil {
		t.Fatal(err)
	}
	if got.Type != "feedback" || got.Reward == nil || *got.Reward != 1 || got.Useful[0] != "m2" {
		t.Fatalf("decoded event = %+v", got)
	}
	if c := stream.Counters(); c.Sent != 2 || c.Dropped != 0 || c.Failed != 0 {
		t.Fatalf("counters = %+v", c)
	}
}

func TestAStalledLogNeverBlocksPublishAndOverflowIsCounted(t *testing.T) {
	producer := &fakeProducer{block: make(chan struct{})}
	stream := NewStream(producer, 4)
	finished := make(chan struct{})
	go func() {
		for range 1000 {
			stream.Publish(memory.Event{Type: "recall", Namespace: "n"})
		}
		close(finished)
	}()
	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("Publish blocked while the log was stalled")
	}
	if c := stream.Counters(); c.Dropped < 990 {
		t.Fatalf("counters = %+v, want almost everything dropped", c)
	}
	close(producer.block)
	stream.Close(context.Background())
	c := stream.Counters()
	if c.Sent+c.Dropped != 1000 {
		t.Fatalf("counters = %+v, want sent + dropped = 1000", c)
	}
}

func TestRejectedRecordsAreCountedAsFailed(t *testing.T) {
	stream := NewStream(&fakeProducer{fail: true}, 8)
	for range 5 {
		stream.Publish(memory.Event{Type: "remember", Namespace: "n"})
	}
	stream.Close(context.Background())
	if c := stream.Counters(); c.Failed != 5 || c.Sent != 0 {
		t.Fatalf("counters = %+v, want 5 failed", c)
	}
}
