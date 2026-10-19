// Package topk keeps the best k items of a stream without sorting the rest.
package topk

import "slices"

// Heap holds at most k items. Its root is the worst item kept, so each new
// item costs one comparison unless it displaces the root.
type Heap[T any] struct {
	items  []T
	k      int
	better func(a, b T) bool
}

// New returns a Heap that keeps the k best items under better, which must be
// a strict total order: ties have to be broken, or which of two equal items
// survives is arbitrary.
func New[T any](k int, better func(a, b T) bool) *Heap[T] {
	return &Heap[T]{items: make([]T, 0, min(k, 256)), k: k, better: better}
}

// Push offers x to the heap.
func (h *Heap[T]) Push(x T) {
	if h.k <= 0 {
		return
	}
	if len(h.items) < h.k {
		h.items = append(h.items, x)
		h.up(len(h.items) - 1)
		return
	}
	if h.better(x, h.items[0]) {
		h.items[0] = x
		h.down(0)
	}
}

// Sorted returns the kept items, best first. The heap must not be used after.
func (h *Heap[T]) Sorted() []T {
	slices.SortFunc(h.items, func(a, b T) int {
		switch {
		case h.better(a, b):
			return -1
		case h.better(b, a):
			return 1
		}
		return 0
	})
	return h.items
}

// up and down keep the worst item at the root.
func (h *Heap[T]) up(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !h.better(h.items[parent], h.items[i]) {
			return
		}
		h.items[parent], h.items[i] = h.items[i], h.items[parent]
		i = parent
	}
}

func (h *Heap[T]) down(i int) {
	n := len(h.items)
	for {
		worst := i
		for _, child := range [2]int{2*i + 1, 2*i + 2} {
			if child < n && h.better(h.items[worst], h.items[child]) {
				worst = child
			}
		}
		if worst == i {
			return
		}
		h.items[i], h.items[worst] = h.items[worst], h.items[i]
		i = worst
	}
}
