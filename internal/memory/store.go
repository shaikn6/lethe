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
	MaxMetadataKeyLen = 128
	MaxMetadataValLen = 1024
	MaxK              = 100

	defaultMaxMemories = 1_000_000
	defaultCompact     = 1000
)

var (
	// ErrInvalid marks a request the caller has to fix.
	ErrInvalid = errors.New("invalid request")
	// ErrNotFound marks an unknown memory or recall.
	ErrNotFound = errors.New("not found")
	// ErrFull means the store holds as many memories as it is allowed to.
	ErrFull = errors.New("store is full")
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
	JournalPath  string // empty keeps everything in memory only
	Fsync        bool   // fsync every journal write
	Candidates   int    // per search method, default 50
	MaxPending   int    // recalls kept waiting for feedback, default 10000
	MaxMemories  int    // live memories across all namespaces, default 1,000,000
	CompactAfter int    // forgets before the journal is rewritten, default 1000
	Seed         uint64 // exploration seed, 0 picks a random one
	Now          func() time.Time

	LearningRate float64 // 0 keeps the policy default
	Temperature  float64 // 0 keeps the policy default
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
	// JournalHealthy is false once the journal file can no longer be
	// written safely; the store then refuses writes instead of losing them.
	JournalHealthy bool `json:"journal_healthy"`
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

func newSpace() *space { return &space{ix: index.New(), byID: make(map[string]uint32)} }

func (sp *space) add(e *entry) {
	sp.byID[e.ID] = sp.ix.Add(e.Text)
	sp.entries = append(sp.entries, e)
	sp.live++
}

type pending struct {
	id        string
	namespace string
	ids       []string
	features  []policy.Features
	slate     []int
	explore   bool
}

// Store is safe for concurrent use. When both locks are needed, dataMu is
// taken before learnMu. Every journal write happens under dataMu, which is
// what lets a compaction take a consistent snapshot.
type Store struct {
	opts Options

	dataMu     sync.RWMutex
	spaces     map[string]*space
	live       int
	tombstones int // forgets since the journal was last rewritten

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
	if opts.MaxMemories <= 0 {
		opts.MaxMemories = defaultMaxMemories
	}
	if opts.CompactAfter <= 0 {
		opts.CompactAfter = defaultCompact
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	seed := opts.Seed
	if seed == 0 {
		seed = mrand.Uint64()
	}
	pol := policy.New()
	if opts.LearningRate > 0 {
		pol.LearningRate = opts.LearningRate
	}
	if opts.Temperature > 0 {
		pol.Temperature = opts.Temperature
	}
	s := &Store{
		opts:    opts,
		spaces:  make(map[string]*space),
		policy:  pol,
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
		// Forgotten memories are still in the file that was just replayed;
		// rewrite it now so they do not outlive this start.
		if s.tombstones > 0 {
			_ = s.compactLocked()
		}
	}
	return s, nil
}

// Close erases forgotten memories from the journal and releases it.
func (s *Store) Close() error {
	if s.journal == nil {
		return nil
	}
	s.dataMu.Lock()
	defer s.dataMu.Unlock()
	if s.tombstones > 0 {
		_ = s.compactLocked()
	}
	return s.journal.close()
}

func (s *Store) replay(rec record) error {
	switch {
	case rec.Remember != nil:
		e, err := s.insert(*rec.Remember)
		if err != nil {
			return err
		}
		e.recalls, e.rewardSum = rec.Recalls, rec.RewardSum
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
	for key, value := range m.Metadata {
		if len(key) > MaxMetadataKeyLen || len(value) > MaxMetadataValLen {
			return Memory{}, fmt.Errorf("%w: metadata keys are limited to %d bytes and values to %d", ErrInvalid, MaxMetadataKeyLen, MaxMetadataValLen)
		}
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
	if s.live >= s.opts.MaxMemories {
		return Memory{}, ErrFull
	}
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
	if _, err := s.insert(m); err != nil {
		return Memory{}, err
	}
	return m, nil
}

// insert adds m to its namespace. The caller holds dataMu or is replaying.
func (s *Store) insert(m Memory) (*entry, error) {
	sp := s.spaces[m.Namespace]
	if sp == nil {
		sp = newSpace()
		s.spaces[m.Namespace] = sp
	}
	if len(m.Embedding) != 0 {
		if sp.dim == 0 {
			sp.dim = len(m.Embedding)
		} else if sp.dim != len(m.Embedding) {
			return nil, fmt.Errorf("%w: embedding has %d dimensions, namespace uses %d", ErrInvalid, len(m.Embedding), sp.dim)
		}
	}
	e := &entry{Memory: m, norm: norm(m.Embedding), alive: true}
	sp.add(e)
	s.live++
	return e, nil
}

// Forget deletes a memory. Its content is dropped from RAM at once and from
// the journal file at the next compaction, which runs after CompactAfter
// forgets, on Close, on the next Open, or when Compact is called.
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
	if s.tombstones >= s.opts.CompactAfter {
		_ = s.compactLocked() // the forget itself is already durable
	}
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
	*sp.entries[doc] = entry{} // drop the text, embedding and metadata
	delete(sp.byID, id)
	sp.live--
	s.live--
	s.tombstones++
}

// Compact rewrites the journal so it holds only live memories and the
// current learning state, and frees what forgotten memories still occupy.
func (s *Store) Compact() error {
	s.dataMu.Lock()
	defer s.dataMu.Unlock()
	return s.compactLocked()
}

// compactLocked requires dataMu. If the rewrite fails the old journal is
// still complete, so the failure is counted and the store keeps running.
func (s *Store) compactLocked() error {
	if s.journal != nil {
		err := s.journal.replace(func(emit func(record) error) error {
			names := make([]string, 0, len(s.spaces))
			for name := range s.spaces {
				names = append(names, name)
			}
			sort.Strings(names)
			for _, name := range names {
				for _, e := range s.spaces[name].entries {
					if !e.alive {
						continue
					}
					if err := emit(record{Remember: &e.Memory, Recalls: e.recalls, RewardSum: e.rewardSum}); err != nil {
						return err
					}
				}
			}
			s.learnMu.Lock()
			defer s.learnMu.Unlock()
			return emit(record{Learn: &learnEntry{Weights: s.policy.Weights, Baseline: s.policy.Baseline, Updates: s.policy.Updates}})
		})
		if err != nil {
			s.journalErrors.Add(1)
			return fmt.Errorf("journal compaction failed: %w", err)
		}
	}
	for name, sp := range s.spaces {
		if sp.live == len(sp.entries) {
			continue
		}
		if sp.live == 0 {
			delete(s.spaces, name)
			continue
		}
		fresh := newSpace()
		fresh.dim = sp.dim
		for _, e := range sp.entries {
			if e.alive {
				fresh.add(e)
			}
		}
		s.spaces[name] = fresh
	}
	s.tombstones = 0
	return nil
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
	defer s.learnMu.Unlock()
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
	for s.order.Len() >= s.opts.MaxPending {
		oldest := s.order.Remove(s.order.Front()).(*pending)
		delete(s.waiting, oldest.id)
	}
	s.waiting[p.id] = s.order.PushBack(p)
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
//
// The learning step and its journal record succeed or fail together: if the
// record cannot be written, the policy is put back as it was and the recall
// stays open, so the caller can retry and RAM never runs ahead of disk.
func (s *Store) Feedback(recallID string, reward float64) error {
	if math.IsNaN(reward) || reward < 0 || reward > 1 {
		return fmt.Errorf("%w: reward must be between 0 and 1", ErrInvalid)
	}
	s.dataMu.Lock()
	defer s.dataMu.Unlock()
	s.learnMu.Lock()
	defer s.learnMu.Unlock()

	el, ok := s.waiting[recallID]
	if !ok {
		return ErrNotFound
	}
	p := el.Value.(*pending)
	before := *s.policy
	if p.explore {
		s.policy.Update(p.features, p.slate, reward)
	}
	ids := make([]string, len(p.slate))
	for i, at := range p.slate {
		ids[i] = p.ids[at]
	}
	if s.journal != nil {
		err := s.journal.append(record{Learn: &learnEntry{
			Weights: s.policy.Weights, Baseline: s.policy.Baseline, Updates: s.policy.Updates,
			Namespace: p.namespace, IDs: ids, Reward: reward,
		}})
		if err != nil {
			*s.policy = before
			s.journalErrors.Add(1)
			return fmt.Errorf("journal write failed: %w", err)
		}
	}
	s.order.Remove(el)
	delete(s.waiting, recallID)
	s.credit(p.namespace, ids, reward)
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

// SetWeights replaces the learned weights, for example to roll back a policy
// that learned from bad feedback or to pin a fixed ranking.
func (s *Store) SetWeights(weights policy.Features) error {
	for _, w := range weights {
		if math.IsNaN(w) || math.Abs(w) > policy.MaxWeight {
			return fmt.Errorf("%w: weights must be finite and within ±%v", ErrInvalid, policy.MaxWeight)
		}
	}
	s.dataMu.Lock()
	defer s.dataMu.Unlock()
	s.learnMu.Lock()
	defer s.learnMu.Unlock()
	if s.journal != nil {
		err := s.journal.append(record{Learn: &learnEntry{Weights: weights, Baseline: 0.5, Updates: s.policy.Updates}})
		if err != nil {
			s.journalErrors.Add(1)
			return fmt.Errorf("journal write failed: %w", err)
		}
	}
	s.policy.Weights = weights
	s.policy.Baseline = 0.5
	return nil
}

// Stats returns a snapshot of the store.
func (s *Store) Stats() Stats {
	st := Stats{JournalHealthy: true}
	s.dataMu.RLock()
	st.Namespaces = len(s.spaces)
	st.Memories = s.live
	s.dataMu.RUnlock()
	s.learnMu.Lock()
	st.Pending = s.order.Len()
	st.Updates = s.policy.Updates
	st.Baseline = s.policy.Baseline
	st.Weights = s.policy.Weights
	s.learnMu.Unlock()
	st.JournalErrors = s.journalErrors.Load()
	if s.journal != nil {
		st.JournalHealthy = s.journal.healthy()
	}
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
