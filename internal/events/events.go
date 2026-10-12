// Package events streams what the memory store does to an outside log such
// as Kafka, for auditing and for training policies offline.
//
// The stream is strictly best effort and can never slow the store down or
// make a request fail: publishing is a non-blocking hand-off to a bounded
// queue, and when the log is slow or down, events are dropped and counted.
// The journal, not this stream, is the source of truth.
package events

import (
	"context"
	"encoding/json"
	"sync"
	"sync/atomic"

	"github.com/shaikn6/lethe/internal/memory"
)

// Producer sends one record to the log and reports the outcome through
// done, which may be called from any goroutine.
type Producer interface {
	Produce(key, value []byte, done func(error))
	// Flush waits until every record handed to Produce has been reported.
	Flush(ctx context.Context) error
	Close()
}

// Counters describes what happened to the events so far.
type Counters struct {
	Sent    uint64 `json:"sent"`
	Dropped uint64 `json:"dropped"` // queue was full
	Failed  uint64 `json:"failed"`  // the log rejected the record
}

// Stream is a bounded queue in front of a Producer.
type Stream struct {
	producer Producer
	queue    chan memory.Event
	worker   sync.WaitGroup
	closing  sync.Once

	sent, dropped, failed atomic.Uint64
}

// NewStream starts a stream that holds at most buffer events in memory.
func NewStream(producer Producer, buffer int) *Stream {
	s := &Stream{producer: producer, queue: make(chan memory.Event, buffer)}
	s.worker.Add(1)
	go func() {
		defer s.worker.Done()
		for event := range s.queue {
			value, err := json.Marshal(event)
			if err != nil {
				s.failed.Add(1)
				continue
			}
			// Keyed by namespace so one user's events stay in order.
			s.producer.Produce([]byte(event.Namespace), value, func(err error) {
				if err != nil {
					s.failed.Add(1)
				} else {
					s.sent.Add(1)
				}
			})
		}
	}()
	return s
}

// Publish queues an event without ever blocking.
func (s *Stream) Publish(event memory.Event) {
	select {
	case s.queue <- event:
	default:
		s.dropped.Add(1)
	}
}

// Counters returns a snapshot of the stream's counters.
func (s *Stream) Counters() Counters {
	return Counters{Sent: s.sent.Load(), Dropped: s.dropped.Load(), Failed: s.failed.Load()}
}

// Close stops accepting events, sends what is queued for as long as ctx
// allows and closes the producer. Publish must not be called after Close.
func (s *Stream) Close(ctx context.Context) error {
	var err error
	s.closing.Do(func() {
		close(s.queue)
		s.worker.Wait()
		err = s.producer.Flush(ctx)
		s.producer.Close()
	})
	return err
}
