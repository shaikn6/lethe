// Package memory is the store: it keeps memories per namespace, finds
// candidates by keyword and by embedding, lets the policy choose which to
// return, and feeds reward back into the policy.
package memory

import (
	"container/list"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	// Exploration sampling has to be seedable so experiments can be repeated;
	// it is not used for anything secret. IDs come from crypto/rand.
	mrand "math/rand/v2" // nosemgrep: go.lang.security.audit.crypto.math_random.math-random-used
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shaikn6/lethe/internal/index"
	"github.com/shaikn6/lethe/internal/policy"
)

// Limits on what a caller may store.
const (
	MaxTextBytes      = 16 << 10
	MaxNamespaceBytes = 128
	MaxEmbeddingDim   = 4096
	MaxMetadata       = 32
	MaxK              = 100
)

var (
	// ErrInvalid marks a request the caller has to fix.
	ErrInvalid = errors.New("invalid request")
	// ErrNotFound marks an unknown memory or recall.
	ErrNotFound = errors.New("not found")
)

// Memory is one stored item.
type Memory struct {
	ID         string            `json:"id"`
	Namespace  string            `json:"namespace"`
	Text       string            `json:"text"`
	Embedding  []float32         `json:"embedding,omitempty"`
	Importance float64           `json:"importance"`
	CreatedAt  time.Time         `json:"created_at"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// Recalled is a memory returned for a query.
type Recalled struct {
	ID         string            `json:"id"`
	Text       string            `json:"text"`
	Importance float64           `json:"importance"`
	CreatedAt  time.Time         `json:"created_at"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	Score      float64           `json:"score"`
}

// Recall is the answer to one query. Its ID is what feedback refers to.
type Recall struct {
	ID       string     `json:"recall_id"`
	Memories []Recalled `json:"memories"`
}

// Query is one recall request.
type Query struct {
	Namespace string
	Text      string
	Embedding []float32
	K         int
	AsOf      time.Time // zero means now
	Explore   bool      // sample from the policy so feedback can teach it
}

// Options configures a Store. The zero value is an in-memory store.
type Options struct {
	JournalPath string // empty keeps everything in memory only
	Fsync       bool   // fsync every journal write
	Candidates  int    // per search method, default 50
	MaxPending  int    // recalls kept waiting for feedback, default 10000
	Seed        uint64 // exploration seed, 0 picks a random one
	Now         func() time.Time
}

// Stats is a snapshot for health checks and monitoring.
type Stats struct {
	Memories      int             `json:"memories"`
	Namespaces    int             `json:"namespaces"`
	Pending       int             `json:"pending_recalls"`
	Updates       int             `json:"policy_updates"`
	Baseline      float64         `json:"reward_baseline"`
	Weights       policy.Features `json:"weights"`
	JournalErrors uint64          `json:"journal_errors"`
}

type entry struct {
	Memory
	norm      float64
	recalls   int
	rewardSum float64
	alive     bool
}

func (e *entry) utility() float64 { return (e.rewardSum + 1) / (float64(e.recalls) + 2) }

type space struct {
	ix      *index.Index
	entries []*entry // by index document number
	byID    map[string]uint32
	dim     int
	live    int
}

type pending struct {
	id        string
	namespace string
	ids       []string
	features  []policy.Features
	slate     []int
	explore   bool
}

// Store is safe for concurrent use. Memory data and learning state have
// separate locks and are never held together.
type Store struct {
	opts Options

	dataMu sync.RWMutex
	spaces map[string]*space

	learnMu sync.Mutex
	policy  *policy.Policy
	rng     *mrand.Rand
	waiting map[string]*list.Element
	order   *list.List // oldest pending recall at the front

	journal       *journal
	journalErrors atomic.Uint64
}

// Open creates a store and, when a journal path is set, replays it.
func Open(opts Options) (*Store, error) {
	if opts.Candidates <= 0 {
		opts.Candidates = 50
	}
	if opts.MaxPending <= 0 {
		opts.MaxPending = 10000
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	seed := opts.Seed
	if seed == 0 {
		seed = mrand.Uint64()
	}
	s := &Store{
		opts:    opts,
		spaces:  make(map[string]*space),
		policy:  policy.New(),
		rng:     mrand.New(mrand.NewPCG(seed, seed^0x9e3779b97f4a7c15)),
		waiting: make(map[string]*list.Element),
		order:   list.New(),
	}
	if opts.JournalPath != "" {
		j, err := openJournal(opts.JournalPath, opts.Fsync, s.replay)
		if err != nil {
			return nil, err
		}
		s.journal = j
	}
	return s, nil
}

// Close releases the journal.
func (s *Store) Close() error {
	if s.journal == nil {
		return nil
	}
	return s.journal.close()
}

func (s *Store) replay(rec record) error {
	switch {
	case rec.Remember != nil:
		return s.insert(*rec.Remember)
	case rec.Forget != nil:
		s.remove(rec.Forget.Namespace, rec.Forget.ID)
	case rec.Learn != nil:
		s.policy.Weights = rec.Learn.Weights
		s.policy.Baseline = rec.Learn.Baseline
		s.policy.Updates = rec.Learn.Updates
		s.credit(rec.Learn.Namespace, rec.Learn.IDs, rec.Learn.Reward)
	}
	return nil
}

// Remember stores a memory and returns it with its ID and timestamp filled in.
func (s *Store) Remember(m Memory) (Memory, error) {
	if err := validate(m.Namespace, m.Text, m.Embedding); err != nil {
		return Memory{}, err
	}
	if len(m.Metadata) > MaxMetadata {
		return Memory{}, fmt.Errorf("%w: at most %d metadata entries", ErrInvalid, MaxMetadata)
	}
	if math.IsNaN(m.Importance) || m.Importance < 0 || m.Importance > 1 {
		return Memory{}, fmt.Errorf("%w: importance must be between 0 and 1", ErrInvalid)
	}
	if m.Text == "" {
		return Memory{}, fmt.Errorf("%w: text is empty", ErrInvalid)
	}
	m.ID = newID()
	if m.CreatedAt.IsZero() {
		m.CreatedAt = s.opts.Now().UTC()
	}

	s.dataMu.Lock()
	defer s.dataMu.Unlock()
	if sp := s.spaces[m.Namespace]; sp != nil && sp.dim != 0 && len(m.Embedding) != 0 && len(m.Embedding) != sp.dim {
		return Memory{}, fmt.Errorf("%w: embedding has %d dimensions, namespace uses %d", ErrInvalid, len(m.Embedding), sp.dim)
	}
	// Journal first: a memory is only acknowledged once it is on disk.
	if s.journal != nil {
		if err := s.journal.append(record{Remember: &m}); err != nil {
			s.journalErrors.Add(1)
			return Memory{}, fmt.Errorf("journal write failed: %w", err)
		}
	}
	if err := s.insert(m); err != nil {
		return Memory{}, err
	}
	return m, nil
}

// insert adds m to its namespace. The caller holds dataMu or is replaying.
func (s *Store) insert(m Memory) error {
	sp := s.spaces[m.Namespace]
	if sp == nil {
		sp = &space{ix: index.New(), byID: make(map[string]uint32)}
		s.spaces[m.Namespace] = sp
	}
	if len(m.Embedding) != 0 {
		if sp.dim == 0 {
			sp.dim = len(m.Embedding)
		} else if sp.dim != len(m.Embedding) {
			return fmt.Errorf("%w: embedding has %d dimensions, namespace uses %d", ErrInvalid, len(m.Embedding), sp.dim)
		}
	}
	doc := sp.ix.Add(m.Text)
	sp.entries = append(sp.entries, &entry{Memory: m, norm: norm(m.Embedding), alive: true})
	sp.byID[m.ID] = doc
	sp.live++
	return nil
}

// Forget deletes a memory.
func (s *Store) Forget(namespace, id string) error {
	s.dataMu.Lock()
	defer s.dataMu.Unlock()
	sp := s.spaces[namespace]
	if sp == nil {
		return ErrNotFound
	}
	if _, ok := sp.byID[id]; !ok {
		return ErrNotFound
	}
	if s.journal != nil {
		if err := s.journal.append(record{Forget: &forgetEntry{Namespace: namespace, ID: id}}); err != nil {
			s.journalErrors.Add(1)
			return fmt.Errorf("journal write failed: %w", err)
		}
	}
	s.remove(namespace, id)
	return nil
}

func (s *Store) remove(namespace, id string) {
	sp := s.spaces[namespace]
	if sp == nil {
		return
	}
	doc, ok := sp.byID[id]
	if !ok {
		return
	}
	sp.ix.Remove(doc)
	sp.entries[doc].alive = false
	sp.entries[doc].Embedding = nil
	delete(sp.byID, id)
	sp.live--
}

// Recall returns up to q.K memories for the query.
func (s *Store) Recall(q Query) (Recall, error) {
	if err := validate(q.Namespace, q.Text, q.Embedding); err != nil {
		return Recall{}, err
	}
	if q.K <= 0 || q.K > MaxK {
		return Recall{}, fmt.Errorf("%w: k must be between 1 and %d", ErrInvalid, MaxK)
	}
	asOf := q.AsOf
	if asOf.IsZero() {
		asOf = s.opts.Now()
	}

	found, features := s.candidates(q, asOf)

	p := &pending{id: newID(), namespace: q.Namespace, features: features, explore: q.Explore}
	p.ids = make([]string, len(found))
	for i, m := range found {
		p.ids[i] = m.ID
	}

	s.learnMu.Lock()
	if q.Explore {
		p.slate = s.policy.Sample(features, q.K, s.rng)
	} else {
		p.slate = s.policy.Rank(features, q.K)
	}
	out := Recall{ID: p.id, Memories: make([]Recalled, len(p.slate))}
	for i, at := range p.slate {
		m := found[at]
		out.Memories[i] = Recalled{
			ID: m.ID, Text: m.Text, Importance: m.Importance, CreatedAt: m.CreatedAt,
			Metadata: m.Metadata, Score: s.policy.Score(features[at]),
		}
	}
	s.waiting[p.id] = s.order.PushBack(p)
	for s.order.Len() > s.opts.MaxPending {
		oldest := s.order.Remove(s.order.Front()).(*pending)
		delete(s.waiting, oldest.id)
	}
	s.learnMu.Unlock()
	return out, nil
}

// candidates gathers the best keyword and semantic matches and describes
// each one for the policy. Either method alone is enough to answer.
func (s *Store) candidates(q Query, asOf time.Time) ([]Memory, []policy.Features) {
	s.dataMu.RLock()
	defer s.dataMu.RUnlock()
	sp := s.spaces[q.Namespace]
	if sp == nil {
		return nil, nil
	}

	keyword := make(map[uint32]float64)
	var best float64
	for _, hit := range sp.ix.Search(q.Text, s.opts.Candidates) {
		keyword[hit.Doc] = hit.Score
		best = math.Max(best, hit.Score)
	}

	semantic := make(map[uint32]bool)
	qNorm := norm(q.Embedding)
	useVectors := qNorm > 0 && len(q.Embedding) == sp.dim
	if useVectors {
		type scored struct {
			doc uint32
			cos float64
		}
		all := make([]scored, 0, sp.live)
		for doc, e := range sp.entries {
			if e.alive && e.norm > 0 {
				all = append(all, scored{uint32(doc), cosine(q.Embedding, e.Embedding, qNorm, e.norm)})
			}
		}
		sort.Slice(all, func(i, j int) bool {
			if all[i].cos != all[j].cos {
				return all[i].cos > all[j].cos
			}
			return all[i].doc < all[j].doc
		})
		for _, sc := range all[:min(s.opts.Candidates, len(all))] {
			semantic[sc.doc] = true
		}
	}

	docs := make([]uint32, 0, len(keyword)+len(semantic))
	for doc := range keyword {
		docs = append(docs, doc)
	}
	for doc := range semantic {
		if _, dup := keyword[doc]; !dup {
			docs = append(docs, doc)
		}
	}
	sort.Slice(docs, func(i, j int) bool { return docs[i] < docs[j] })

	temporal := 0.0
	if asksAboutTime(q.Text) {
		temporal = 1
	}
	found := make([]Memory, len(docs))
	features := make([]policy.Features, len(docs))
	for i, doc := range docs {
		e := sp.entries[doc]
		found[i] = e.Memory
		found[i].Embedding = nil
		f := &features[i]
		if best > 0 {
			f[policy.Keyword] = keyword[doc] / best
		}
		if useVectors && e.norm > 0 {
			f[policy.Semantic] = math.Max(0, cosine(q.Embedding, e.Embedding, qNorm, e.norm))
		}
		ageDays := math.Max(0, asOf.Sub(e.CreatedAt).Hours()/24)
		f[policy.Recency] = 1 / (1 + ageDays)
		f[policy.Utility] = e.utility()
		f[policy.Importance] = e.Importance
		f[policy.TemporalRecency] = temporal * f[policy.Recency]
		if _, inKeyword := keyword[doc]; inKeyword && semantic[doc] {
			f[policy.Agreement] = 1
		}
	}
	return found, features
}

// Feedback reports how useful a recall was, from 0 (useless) to 1 (it
// answered the question). Each recall accepts feedback once.
func (s *Store) Feedback(recallID string, reward float64) error {
	if math.IsNaN(reward) || reward < 0 || reward > 1 {
		return fmt.Errorf("%w: reward must be between 0 and 1", ErrInvalid)
	}
	s.learnMu.Lock()
	el, ok := s.waiting[recallID]
	if !ok {
		s.learnMu.Unlock()
		return ErrNotFound
	}
	p := s.order.Remove(el).(*pending)
	delete(s.waiting, recallID)
	if p.explore {
		s.policy.Update(p.features, p.slate, reward)
	}
	learned := &learnEntry{
		Weights: s.policy.Weights, Baseline: s.policy.Baseline, Updates: s.policy.Updates,
		Namespace: p.namespace, Reward: reward,
	}
	s.learnMu.Unlock()

	for _, at := range p.slate {
		learned.IDs = append(learned.IDs, p.ids[at])
	}
	s.dataMu.Lock()
	s.credit(p.namespace, learned.IDs, reward)
	s.dataMu.Unlock()

	// Losing a learning record costs one step of learning, not data, so a
	// journal failure here is counted instead of failing the request.
	if s.journal != nil {
		if err := s.journal.append(record{Learn: learned}); err != nil {
			s.journalErrors.Add(1)
		}
	}
	return nil
}

// credit records that the given memories were in a slate that earned reward.
func (s *Store) credit(namespace string, ids []string, reward float64) {
	sp := s.spaces[namespace]
	if sp == nil {
		return
	}
	for _, id := range ids {
		if doc, ok := sp.byID[id]; ok {
			sp.entries[doc].recalls++
			sp.entries[doc].rewardSum += reward
		}
	}
}

// Stats returns a snapshot of the store.
func (s *Store) Stats() Stats {
	var st Stats
	s.dataMu.RLock()
	st.Namespaces = len(s.spaces)
	for _, sp := range s.spaces {
		st.Memories += sp.live
	}
	s.dataMu.RUnlock()
	s.learnMu.Lock()
	st.Pending = s.order.Len()
	st.Updates = s.policy.Updates
	st.Baseline = s.policy.Baseline
	st.Weights = s.policy.Weights
	s.learnMu.Unlock()
	st.JournalErrors = s.journalErrors.Load()
	return st
}

func validate(namespace, text string, embedding []float32) error {
	if namespace == "" || len(namespace) > MaxNamespaceBytes {
		return fmt.Errorf("%w: namespace must be 1 to %d bytes", ErrInvalid, MaxNamespaceBytes)
	}
	if len(text) > MaxTextBytes {
		return fmt.Errorf("%w: text is longer than %d bytes", ErrInvalid, MaxTextBytes)
	}
	if len(embedding) > MaxEmbeddingDim {
		return fmt.Errorf("%w: embedding has more than %d dimensions", ErrInvalid, MaxEmbeddingDim)
	}
	for _, v := range embedding {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("%w: embedding contains a non-finite value", ErrInvalid)
		}
	}
	return nil
}

var timeWords = map[string]struct{}{}

func init() {
	for _, w := range []string{
		"when", "last", "latest", "recent", "recently", "newest", "current", "currently", "now",
		"today", "yesterday", "tomorrow", "ago", "date", "first", "earliest", "before", "after", "previous",
	} {
		timeWords[w] = struct{}{}
	}
}

func asksAboutTime(query string) bool {
	for _, token := range index.Tokenize(query) {
		if _, ok := timeWords[token]; ok {
			return true
		}
	}
	return false
}

func norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func cosine(a, b []float32, aNorm, bNorm float64) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot / (aNorm * bNorm)
}

func newID() string {
	var raw [12]byte
	if _, err := rand.Read(raw[:]); err != nil {
		panic(err) // the system random source is gone; nothing sensible to do
	}
	return hex.EncodeToString(raw[:])
}
