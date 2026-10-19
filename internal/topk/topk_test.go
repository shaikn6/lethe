package topk

import (
	"math/rand/v2"
	"slices"
	"testing"
)

type item struct {
	id    int
	score int
}

func better(a, b item) bool {
	if a.score != b.score {
		return a.score > b.score
	}
	return a.id < b.id
}

func TestMatchesFullSort(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{0, 1, 5, 50, 1000} {
		for _, k := range []int{0, 1, 5, 50, 2000} {
			items := make([]item, n)
			for i := range items {
				items[i] = item{id: i, score: rng.IntN(10)} // many ties
			}
			want := slices.Clone(items)
			slices.SortFunc(want, func(a, b item) int {
				if better(a, b) {
					return -1
				}
				if better(b, a) {
					return 1
				}
				return 0
			})
			want = want[:min(k, n)]

			h := New(k, better)
			for _, it := range items {
				h.Push(it)
			}
			got := h.Sorted()
			if len(got) != len(want) || (len(want) > 0 && !slices.Equal(got, want)) {
				t.Fatalf("n=%d k=%d: got %v, want %v", n, k, got, want)
			}
		}
	}
}
