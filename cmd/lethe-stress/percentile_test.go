package main

import (
	"testing"
	"time"
)

func TestPercentilePicksTheNearestRank(t *testing.T) {
	five := []time.Duration{10, 20, 30, 40, 50} // ascending, len-1 == 4
	cases := []struct {
		name   string
		sorted []time.Duration
		p      float64
		want   time.Duration
	}{
		{"empty is zero", nil, 0.5, 0},
		{"single element any p", []time.Duration{7}, 0.99, 7},
		{"min", five, 0, 10},
		{"median", five, 0.50, 30},
		{"p99 rounds down to an index that exists", five, 0.99, 40},
		{"max never runs off the end", five, 1.0, 50},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := percentile(c.sorted, c.p); got != c.want {
				t.Errorf("percentile(%v, %v) = %v, want %v", c.sorted, c.p, got, c.want)
			}
		})
	}
}
