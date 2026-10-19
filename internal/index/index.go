// Package index is a small in-memory BM25 index that supports adding and
// removing documents.
package index

import (
	"math"
	"strings"
	"unicode"

	"github.com/shaikn6/lethe/internal/topk"
)

const (
	k1 = 1.2
	b  = 0.75
)

// Scored is a document number with its BM25 score.
type Scored struct {
	Doc   uint32
	Score float64
}

type posting struct {
	doc uint32
	tf  uint16
}

// Index is not safe for concurrent use; callers serialize access.
type Index struct {
	postings map[string][]posting // may hold removed documents
	df       map[string]int       // counts live documents only
	terms    [][]string           // distinct terms of each document
	length   []uint32
	alive    []bool
	live     int
	total    uint64
}

var stopwords = map[string]struct{}{}

func init() {
	for _, word := range strings.Fields("a an and are as at be by for from has he in is it its of on or that the to was were will with") {
		stopwords[word] = struct{}{}
	}
}

// Tokenize lowercases s, splits it on anything that is not a letter or digit
// and drops stopwords.
func Tokenize(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	tokens := fields[:0]
	for _, field := range fields {
		if _, stop := stopwords[field]; !stop {
			tokens = append(tokens, field)
		}
	}
	return tokens
}

// New returns an empty index.
func New() *Index {
	return &Index{postings: make(map[string][]posting), df: make(map[string]int)}
}

// Len reports the number of live documents.
func (ix *Index) Len() int { return ix.live }

// Add indexes text and returns its document number.
func (ix *Index) Add(text string) uint32 {
	doc := uint32(len(ix.alive))
	tokens := Tokenize(text)
	counts := make(map[string]uint16, len(tokens))
	for _, token := range tokens {
		if counts[token] < math.MaxUint16 {
			counts[token]++
		}
	}
	distinct := make([]string, 0, len(counts))
	for token, tf := range counts {
		ix.postings[token] = append(ix.postings[token], posting{doc: doc, tf: tf})
		ix.df[token]++
		distinct = append(distinct, token)
	}
	ix.terms = append(ix.terms, distinct)
	ix.length = append(ix.length, uint32(len(tokens)))
	ix.alive = append(ix.alive, true)
	ix.live++
	ix.total += uint64(len(tokens))
	return doc
}

// Remove hides a document from future searches. Unknown or already removed
// documents are ignored.
func (ix *Index) Remove(doc uint32) {
	if int(doc) >= len(ix.alive) || !ix.alive[doc] {
		return
	}
	ix.alive[doc] = false
	ix.live--
	ix.total -= uint64(ix.length[doc])
	for _, token := range ix.terms[doc] {
		ix.df[token]--
		// A word that only removed documents used must not linger as a key.
		if ix.df[token] == 0 {
			delete(ix.df, token)
			delete(ix.postings, token)
		}
	}
	ix.terms[doc] = nil
}

// Search returns up to n live documents for query, best first. Ties are
// broken by document number so results are deterministic.
func (ix *Index) Search(query string, n int) []Scored {
	if n <= 0 || ix.live == 0 {
		return nil
	}
	avg := float64(ix.total) / float64(ix.live)
	if avg == 0 {
		return nil
	}
	scores := make(map[uint32]float64)
	seen := make(map[string]struct{})
	for _, term := range Tokenize(query) {
		if _, dup := seen[term]; dup {
			continue
		}
		seen[term] = struct{}{}
		df := float64(ix.df[term])
		if df == 0 {
			continue
		}
		idf := math.Log(1 + (float64(ix.live)-df+0.5)/(df+0.5))
		for _, p := range ix.postings[term] {
			if !ix.alive[p.doc] {
				continue
			}
			tf := float64(p.tf)
			norm := k1 * (1 - b + b*float64(ix.length[p.doc])/avg)
			scores[p.doc] += idf * tf * (k1 + 1) / (tf + norm)
		}
	}
	// Only the best n are wanted, so keep those instead of sorting every hit.
	top := topk.New(n, func(a, b Scored) bool {
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		return a.Doc < b.Doc
	})
	for doc, score := range scores {
		top.Push(Scored{Doc: doc, Score: score})
	}
	return top.Sorted()
}
