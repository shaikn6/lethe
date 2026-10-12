package policy

import (
	"math"
	"math/rand/v2"
	"testing"
)

func feat(keyword, semantic, recency float64) Features {
	var f Features
	f[Keyword], f[Semantic], f[Recency] = keyword, semantic, recency
	return f
}

func TestRankOrdersByScoreAndLimitsK(t *testing.T) {
	p := New()
	cands := []Features{feat(0.1, 0.1, 0), feat(0.9, 0.8, 0), feat(0.5, 0.5, 0)}
	got := p.Rank(cands, 2)
	if len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("Rank = %v, want [1 2]", got)
	}
	if got := p.Rank(cands, 10); len(got) != 3 {
		t.Fatalf("Rank with large k returned %d items", len(got))
	}
	if got := p.Rank(nil, 3); len(got) != 0 {
		t.Fatalf("Rank(nil) = %v", got)
	}
}

func TestSampleReturnsDistinctItemsAndPrefersHighScores(t *testing.T) {
	p := New()
	rng := rand.New(rand.NewPCG(1, 2))
	cands := []Features{feat(1, 1, 0), feat(0, 0, 0), feat(0, 0, 0), feat(0, 0, 0)}
	first := 0
	for range 2000 {
		slate := p.Sample(cands, 2, rng)
		if len(slate) != 2 || slate[0] == slate[1] {
			t.Fatalf("slate = %v, want two distinct items", slate)
		}
		if slate[0] == 0 {
			first++
		}
	}
	if first < 1400 {
		t.Fatalf("best item sampled first %d/2000 times, want a clear majority", first)
	}
}

func TestUpdateIgnoresInvalidInput(t *testing.T) {
	p := New()
	before := p.Weights
	cands := []Features{feat(1, 0, 0), feat(0, 1, 0)}
	p.Update(cands, []int{5}, 1, nil)
	p.Update(cands, nil, 1, nil)
	p.Update(cands, []int{0}, math.NaN(), nil)
	p.Update(cands, []int{0}, 1, []float64{1, 0})
	p.Update(cands, []int{0}, 1, []float64{math.Inf(1)})
	if p.Weights != before {
		t.Fatalf("weights changed on invalid input: %v", p.Weights)
	}
}

func TestWeightsStayFinite(t *testing.T) {
	p := New()
	rng := rand.New(rand.NewPCG(3, 4))
	cands := []Features{feat(1, 1, 1), feat(0, 0, 0)}
	for i := range 5000 {
		p.Update(cands, p.Sample(cands, 1, rng), float64(i%2)*1000, nil)
	}
	for i, w := range p.Weights {
		if math.IsNaN(w) || math.IsInf(w, 0) || math.Abs(w) > MaxWeight {
			t.Fatalf("weight %d = %v, want finite and within ±%v", i, w, MaxWeight)
		}
	}
}

// The useful memory is the most recent one; keyword and semantic scores are
// noise. A policy that starts out trusting only those two has to discover
// recency from reward alone.
func TestLearnsWhichSignalPredictsUsefulMemories(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	episode := func() ([]Features, int) {
		cands := make([]Features, 20)
		best := 0
		for i := range cands {
			cands[i] = feat(rng.Float64(), rng.Float64(), rng.Float64())
			if cands[i][Recency] > cands[best][Recency] {
				best = i
			}
		}
		return cands, best
	}
	hitRate := func(p *Policy) float64 {
		hits := 0
		for range 500 {
			cands, best := episode()
			for _, i := range p.Rank(cands, 3) {
				if i == best {
					hits++
				}
			}
		}
		return float64(hits) / 500
	}

	p := New()
	before := hitRate(p)
	for range 4000 {
		cands, best := episode()
		slate := p.Sample(cands, 3, rng)
		reward := 0.0
		for _, i := range slate {
			if i == best {
				reward = 1
			}
		}
		p.Update(cands, slate, reward, nil)
	}
	after := hitRate(p)
	t.Logf("hit rate before %.2f, after %.2f, weights %v", before, after, p.Weights)
	if before > 0.35 {
		t.Fatalf("untrained hit rate %.2f is too high for the test to mean anything", before)
	}
	if after < 0.85 {
		t.Fatalf("hit rate after training = %.2f, want >= 0.85", after)
	}
}

// A candidate the starting weights score at zero is useful. Without forced
// exploration it would almost never be tried.
func TestExplorationFindsCandidatesTheCurrentWeightsIgnore(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 9))
	episode := func() ([]Features, int) {
		cands := make([]Features, 30)
		for i := range cands {
			cands[i] = feat(0.5+rng.Float64()/2, 0.5+rng.Float64()/2, 0)
		}
		hidden := rng.IntN(len(cands))
		cands[hidden] = Features{}
		cands[hidden][NeighborSemantic] = 1
		return cands, hidden
	}
	for _, perItem := range []bool{false, true} {
		p := New()
		for range 6000 {
			cands, hidden := episode()
			slate := p.Sample(cands, 5, rng)
			reward, items := 0.0, make([]float64, len(slate))
			for i, pick := range slate {
				if pick == hidden {
					reward, items[i] = 1, 1
				}
			}
			if !perItem {
				items = nil
			}
			p.Update(cands, slate, reward, items)
		}
		hits := 0
		for range 300 {
			cands, hidden := episode()
			if p.Rank(cands, 1)[0] == hidden {
				hits++
			}
		}
		t.Logf("per-item=%v: hidden candidate ranked first %d/300, neighbour weight %.2f", perItem, hits, p.Weights[NeighborSemantic])
		if hits < 270 {
			t.Errorf("per-item=%v: hidden candidate ranked first only %d/300 times", perItem, hits)
		}
	}
}
