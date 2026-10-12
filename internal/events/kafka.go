package events

import (
	"context"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"
)

type kafkaProducer struct {
	client *kgo.Client
}

// NewKafka returns a Producer that writes to topic on the given brokers.
// It connects lazily, so a broker that is down at start does not stop the
// server; records simply fail until the broker is back.
func NewKafka(brokers []string, topic string) (Producer, error) {
	client, err := kgo.NewClient(
		kgo.SeedBrokers(brokers...),
		kgo.DefaultProduceTopic(topic),
		kgo.ProducerLinger(20*time.Millisecond),
		kgo.RecordDeliveryTimeout(15*time.Second),
		kgo.MaxBufferedRecords(10_000),
	)
	if err != nil {
		return nil, err
	}
	return &kafkaProducer{client: client}, nil
}

// Produce never blocks: when the client's own buffer is full the record is
// reported as failed straight away.
func (k *kafkaProducer) Produce(key, value []byte, done func(error)) {
	k.client.TryProduce(context.Background(), &kgo.Record{Key: key, Value: value}, func(_ *kgo.Record, err error) {
		done(err)
	})
}

func (k *kafkaProducer) Flush(ctx context.Context) error { return k.client.Flush(ctx) }

func (k *kafkaProducer) Close() { k.client.Close() }
