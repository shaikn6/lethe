// Package policy ranks candidate memories and learns the ranking from reward.
//
// The policy is a Plackett-Luce model over a linear score: a slate is drawn
// one item at a time, each with probability proportional to exp(score/T)
// among the items not yet chosen. It is trained with REINFORCE: after a
// slate is used, the reward it earned moves the weights along the gradient
// of that slate's log-probability.
package policy

import (
	"math"
	"math/rand/v2" // nosemgrep: go.lang.security.audit.crypto.math_random.math-random-used -- seedable sampling, nothing secret
	"sort"
)

// Feature positions.
const (
	Keyword         = iota // BM25 score, scaled to the best candidate
	Semantic               // embedding cosine similarity
	Recency                // newer memories score higher
	Utility                // how often this memory was in a rewarded slate
	Importance             // set by the writer
	TemporalRecency        // recency, only when the query asks about time
	Agreement              // found by both keyword and semantic search
	NumFeatures
)

// MaxWeight bounds every weight so a burst of extreme rewards cannot push
// the policy into a state it never recovers from.
const MaxWeight = 50.0

// Features describes one candidate for one query. Values are expected in [0, 1].
type Features [NumFeatures]float64

// Policy holds the learned weights. It is not safe for concurrent use.
type Policy struct {
	Weights      Features
	Baseline     float64 // running mean reward
	Updates      int
	LearningRate float64
	Temperature  float64
}

// New returns a policy that starts as an even mix of keyword and semantic
// search, which is the usual hand-built hybrid.
func New() *Policy {
	p := &Policy{Baseline: 0.5, LearningRate: 0.2, Temperature: 0.5}
	p.Weights[Keyword] = 2
	p.Weights[Semantic] = 2
	return p
}

// Score is the linear score of one candidate.
func (p *Policy) Score(f Features) float64 {
	var s float64
	for i, w := range p.Weights {
		s += w * f[i]
	}
	return s
}

// Rank returns the indexes of the k best candidates, best first.
func (p *Policy) Rank(cands []Features, k int) []int {
	order := make([]int, len(cands))
	scores := make([]float64, len(cands))
	for i, f := range cands {
		order[i], scores[i] = i, p.Score(f)
	}
	sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })
	return order[:min(k, len(order))]
}

// Sample draws a slate of up to k distinct candidates from the policy.
func (p *Policy) Sample(cands []Features, k int, rng *rand.Rand) []int {
	k = min(k, len(cands))
	remaining := make([]int, len(cands))
	for i := range remaining {
		remaining[i] = i
	}
	slate := make([]int, 0, max(k, 0))
	for len(slate) < k {
		probs := p.probabilities(cands, remaining)
		x, pick := rng.Float64(), len(remaining)-1
		for i, pr := range probs {
			if x < pr {
				pick = i
				break
			}
			x -= pr
		}
		slate = append(slate, remaining[pick])
		remaining = append(remaining[:pick], remaining[pick+1:]...)
	}
	return slate
}

// Update applies one REINFORCE step for a slate that earned reward. Slates
// that refer to unknown candidates and non-finite rewards are ignored.
func (p *Policy) Update(cands []Features, slate []int, reward float64) {
	if len(slate) == 0 || math.IsNaN(reward) || math.IsInf(reward, 0) {
		return
	}
	chosen := make(map[int]bool, len(slate))
	for _, i := range slate {
		if i < 0 || i >= len(cands) || chosen[i] {
			return
		}
		chosen[i] = true
	}
	reward = math.Max(0, math.Min(1, reward))

	remaining := make([]int, len(cands))
	for i := range remaining {
		remaining[i] = i
	}
	var grad Features
	for _, pick := range slate {
		probs := p.probabilities(cands, remaining)
		at := 0
		for i, cand := range remaining {
			if cand == pick {
				at = i
			}
			for j := range grad {
				grad[j] -= probs[i] * cands[cand][j]
			}
		}
		for j := range grad {
			grad[j] += cands[pick][j]
		}
		remaining = append(remaining[:at], remaining[at+1:]...)
	}

	step := p.LearningRate * (reward - p.Baseline) / p.Temperature
	for j := range p.Weights {
		p.Weights[j] = math.Max(-MaxWeight, math.Min(MaxWeight, p.Weights[j]+step*grad[j]))
	}
	p.Baseline += 0.05 * (reward - p.Baseline)
	p.Updates++
}

// probabilities is the softmax of score/T over the remaining candidates.
func (p *Policy) probabilities(cands []Features, remaining []int) []float64 {
	probs := make([]float64, len(remaining))
	top := math.Inf(-1)
	for i, cand := range remaining {
		probs[i] = p.Score(cands[cand]) / p.Temperature
		top = math.Max(top, probs[i])
	}
	var sum float64
	for i := range probs {
		probs[i] = math.Exp(probs[i] - top)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}
