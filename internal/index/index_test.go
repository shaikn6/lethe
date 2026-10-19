package index

import (
	"fmt"
	"reflect"
	"slices"
	"testing"
)

func docs(hits []Scored) []uint32 {
	out := make([]uint32, len(hits))
	for i, hit := range hits {
		out[i] = hit.Doc
	}
	return out
}

func TestTokenize(t *testing.T) {
	got := Tokenize("The dentist appointment — moved to 3pm, on Friday!")
	want := []string{"dentist", "appointment", "moved", "3pm", "friday"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Tokenize = %v, want %v", got, want)
	}
}

func TestSearchRanksBestMatchFirst(t *testing.T) {
	ix := New()
	ix.Add("Maria adopted a beagle named Biscuit")
	target := ix.Add("Maria's beagle Biscuit loves the dog park on Sundays")
	ix.Add("The office moved to a new building downtown")

	hits := ix.Search("beagle dog park", 10)
	if len(hits) != 2 || hits[0].Doc != target {
		t.Fatalf("hits = %+v, want doc %d first of two", hits, target)
	}
	if hits[0].Score <= hits[1].Score {
		t.Fatalf("scores not descending: %+v", hits)
	}
}

func TestRareTermOutweighsCommonTerm(t *testing.T) {
	ix := New()
	ix.Add("meeting notes from monday")
	ix.Add("meeting notes from tuesday")
	rare := ix.Add("meeting about the kyoto trip")
	if hits := ix.Search("meeting kyoto", 1); hits[0].Doc != rare {
		t.Fatalf("top = %d, want %d", hits[0].Doc, rare)
	}
}

func TestRemoveHidesDocumentAndRestoresStatistics(t *testing.T) {
	ix := New()
	a := ix.Add("alpha beta")
	b := ix.Add("alpha gamma")
	before := ix.Search("gamma", 5)

	c := ix.Add("alpha delta gamma gamma")
	ix.Remove(c)
	ix.Remove(c) // removing twice is a no-op

	if got := docs(ix.Search("delta", 5)); len(got) != 0 {
		t.Fatalf("removed document still found: %v", got)
	}
	if after := ix.Search("gamma", 5); !reflect.DeepEqual(before, after) {
		t.Fatalf("statistics not restored:\n%+v\n%+v", before, after)
	}
	if got := docs(ix.Search("alpha", 5)); !reflect.DeepEqual(got, []uint32{a, b}) {
		t.Fatalf("alpha hits = %v, want [%d %d]", got, a, b)
	}
	if ix.Len() != 2 {
		t.Fatalf("Len = %d, want 2", ix.Len())
	}
}

func TestSearchEdgeCases(t *testing.T) {
	ix := New()
	if hits := ix.Search("anything", 5); len(hits) != 0 {
		t.Fatalf("empty index returned %+v", hits)
	}
	ix.Add("one two three")
	ix.Add("one two")
	ix.Add("one")
	if got := len(ix.Search("one", 2)); got != 2 {
		t.Fatalf("len = %d, want 2", got)
	}
	for _, query := range []string{"", "the of and", "zzz"} {
		if hits := ix.Search(query, 5); len(hits) != 0 {
			t.Fatalf("Search(%q) = %+v, want none", query, hits)
		}
	}
	if !reflect.DeepEqual(ix.Search("one two", 3), ix.Search("two one one", 3)) {
		t.Fatal("term order or repetition changed the result")
	}
}

func TestSearchLimitIsPrefixOfFullRanking(t *testing.T) {
	ix := New()
	for i := range 300 {
		ix.Add(fmt.Sprintf("deploy service %d status %d host %d", i%17, i%5, i%3)) // many equal scores
	}
	full := ix.Search("deploy service status", 1000)
	for _, n := range []int{1, 7, 50, 299, 300} {
		got := ix.Search("deploy service status", n)
		if !slices.Equal(got, full[:n]) {
			t.Fatalf("n=%d: top results differ from the full ranking", n)
		}
	}
}
