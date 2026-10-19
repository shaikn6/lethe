package memory

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// benchStore fills one namespace with n memories that carry both text and a
// dim-sized embedding, so a recall exercises the keyword and semantic paths.
func benchStore(b *testing.B, n, dim int) *Store {
	b.Helper()
	s, err := Open(Options{Seed: 1})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { s.Close() })
	rng := rand.New(rand.NewPCG(7, 11))
	for i := range n {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32() - 0.5
		}
		_, err := s.Remember(Memory{
			Namespace: "bench",
			Text:      fmt.Sprintf("turn %d: the deploy of service %d finished with status %d on host %d", i, i%50, i%7, i%113),
			Embedding: vec,
		})
		if err != nil {
			b.Fatal(err)
		}
	}
	return s
}

func BenchmarkRecall(b *testing.B) {
	const dim = 64
	for _, n := range []int{1_000, 10_000, 50_000} {
		b.Run(fmt.Sprintf("memories=%d", n), func(b *testing.B) {
			s := benchStore(b, n, dim)
			rng := rand.New(rand.NewPCG(3, 5))
			query := make([]float32, dim)
			for j := range query {
				query[j] = rng.Float32() - 0.5
			}
			q := Query{Namespace: "bench", Text: "deploy of service 7 status", Embedding: query, K: 5}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := s.Recall(q); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
