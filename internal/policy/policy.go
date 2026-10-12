// Package policy ranks candidate memories and learns the ranking from reward.
//
// The policy is a Plackett-Luce model over a linear score: a slate is drawn
// one item at a time, each with probability proportional to exp(score/T)
// among the items not yet chosen. With a small probability a slot is filled
// uniformly at random instead, so that candidates the current weights score
// at zero still get tried and can earn reward.
//
// It is trained with REINFORCE: after a slate is used, the reward it earned
// moves the weights along the gradient of the slate's log-probability. The
// reward can be one number for the whole slate or one per returned item;
// per-item reward gives a much less noisy gradient. Steps are scaled with
// Adam so that progress does not depend on the size of the raw gradient.
package policy

import (
	"math"
	"math/rand/v2" // nosemgrep: go.lang.security.audit.crypto.math_random.math-random-used -- seedable sampling, nothing secret
	"sort"
)

// Feature positions.
const (
	Keyword          = iota // BM25 score, scaled to the best candidate
	Semantic                // embedding cosine similarity
	Recency                 // newer memories score higher
	Utility                 // how often this memory was in a rewarded slate
	Importance              // set by the writer
	TemporalRecency         // recency, only when the query asks about time
	Agreement               // found by both keyword and semantic search
	NeighborKeyword         // keyword score of the memories stored right before and after
	NeighborSemantic        // semantic score of the memories stored right before and after
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
	Baseline     float64 // running mean of slate reward
	ItemBaseline float64 // running mean of per-item reward
	Updates      int
	LearningRate float64
	Temperature  float64
	Epsilon      float64 // chance that a slot is filled uniformly at random

	m, v Features // Adam moment estimates; rebuilt after a restart
}

const (
	adamBeta1 = 0.9
	adamBeta2 = 0.999
	adamEps   = 1e-8
)

// New returns a policy that starts as an even mix of keyword and semantic
// search, which is the usual hand-built hybrid.
func New() *Policy {
	p := &Policy{Baseline: 0.5, LearningRate: 0.02, Temperature: 0.5, Epsilon: 0.1}
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
		pick := len(remaining) - 1
		if rng.Float64() < p.Epsilon {
			pick = rng.IntN(len(remaining))
		} else {
			x := rng.Float64()
			for i, pr := range p.probabilities(cands, remaining) {
				if x < pr {
					pick = i
					break
				}
				x -= pr
			}
		}
		slate = append(slate, remaining[pick])
		remaining = append(remaining[:pick], remaining[pick+1:]...)
	}
	return slate
}

// Update applies one learning step for a slate that earned reward. When
// itemRewards has one value per slate position, each position is credited
// with its own reward; otherwise every position shares reward. Slates that
// refer to unknown candidates and non-finite rewards are ignored.
func (p *Policy) Update(cands []Features, slate []int, reward float64, itemRewards []float64) {
	if len(slate) == 0 || math.IsNaN(reward) || math.IsInf(reward, 0) {
		return
	}
	if itemRewards != nil && len(itemRewards) != len(slate) {
		return
	}
	chosen := make(map[int]bool, len(slate))
	for _, i := range slate {
		if i < 0 || i >= len(cands) || chosen[i] {
			return
		}
		chosen[i] = true
	}
	for _, r := range itemRewards {
		if math.IsNaN(r) || math.IsInf(r, 0) {
			return
		}
	}
	reward = clamp01(reward)

	remaining := make([]int, len(cands))
	for i := range remaining {
		remaining[i] = i
	}
	var grad Features
	var itemMean float64
	for slot, pick := range slate {
		probs := p.probabilities(cands, remaining)
		at := 0
		var expected Features
		for i, cand := range remaining {
			if cand == pick {
				at = i
			}
			for j := range expected {
				expected[j] += probs[i] * cands[cand][j]
			}
		}
		// The slot was drawn from a mix of the policy and a uniform pick, so
		// the policy's share of the probability scales its gradient.
		fromPolicy := (1 - p.Epsilon) * probs[at]
		share := fromPolicy / (fromPolicy + p.Epsilon/float64(len(remaining)))
		advantage := reward - p.Baseline
		if itemRewards != nil {
			r := clamp01(itemRewards[slot])
			advantage = r - p.ItemBaseline
			itemMean += r / float64(len(slate))
		}
		for j := range grad {
			grad[j] += advantage * share * (cands[pick][j] - expected[j]) / p.Temperature
		}
		remaining = append(remaining[:at], remaining[at+1:]...)
	}

	p.Updates++
	for j := range p.Weights {
		p.m[j] = adamBeta1*p.m[j] + (1-adamBeta1)*grad[j]
		p.v[j] = adamBeta2*p.v[j] + (1-adamBeta2)*grad[j]*grad[j]
		mHat := p.m[j] / (1 - math.Pow(adamBeta1, float64(p.Updates)))
		vHat := p.v[j] / (1 - math.Pow(adamBeta2, float64(p.Updates)))
		step := p.LearningRate * mHat / (math.Sqrt(vHat) + adamEps)
		p.Weights[j] = math.Max(-MaxWeight, math.Min(MaxWeight, p.Weights[j]+step))
	}
	p.Baseline += 0.05 * (reward - p.Baseline)
	if itemRewards != nil {
		p.ItemBaseline += 0.05 * (itemMean - p.ItemBaseline)
	}
}

func clamp01(x float64) float64 { return math.Max(0, math.Min(1, x)) }

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
