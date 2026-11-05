// Package server is the HTTP API in front of the memory store.
package server

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/shaikn6/lethe/internal/memory"
	"github.com/shaikn6/lethe/internal/policy"
)

const (
	maxBodyBytes = 1 << 20
	defaultK     = 5
)

type rememberRequest struct {
	Namespace  string            `json:"namespace"`
	Text       string            `json:"text"`
	Embedding  []float32         `json:"embedding"`
	Importance *float64          `json:"importance"`
	CreatedAt  time.Time         `json:"created_at"`
	Metadata   map[string]string `json:"metadata"`
}

type recallRequest struct {
	Namespace string    `json:"namespace"`
	Query     string    `json:"query"`
	Embedding []float32 `json:"embedding"`
	K         int       `json:"k"`
	AsOf      time.Time `json:"as_of"`
	Explore   bool      `json:"explore"`
}

type policyRequest struct {
	Weights []float64 `json:"weights"`
}

type feedbackRequest struct {
	RecallID string   `json:"recall_id"`
	Reward   *float64 `json:"reward"`
	Useful   []string `json:"useful"` // optional: ids of the returned memories that helped
}

// Config holds the settings of the HTTP API.
type Config struct {
	// Token protects the memory endpoints. Empty leaves them open, which is
	// only reasonable on a loopback address.
	Token string
	// AdminToken protects the endpoints that change how the server behaves
	// for everyone: setting the ranking weights and forcing a compaction.
	// Empty switches those endpoints off.
	AdminToken string
	// Scopes optionally limits data tokens to a set of namespaces. Each key is
	// a bearer token and its value is the namespaces that token may touch; a
	// value of {"*"} grants every namespace. When Scopes is empty the data
	// endpoints behave exactly as before: Token alone guards them and grants
	// all namespaces. See ParseScopes for the LETHE_TOKEN_SCOPES format.
	Scopes map[string][]string
	// MaxInFlight is how many requests are served at once before the rest
	// are turned away with 503. Default 256.
	MaxInFlight int
	// CompactEvery is the shortest gap between compactions requested over
	// the API, because each one briefly holds every other request. Default 30s.
	CompactEvery time.Duration
	// EventCounters, when set, is reported by the health check so that a
	// stalled event log is visible.
	EventCounters func() any
}

// New returns the API handler.
func New(store *memory.Store, cfg Config, log *slog.Logger) http.Handler {
	if cfg.MaxInFlight <= 0 {
		cfg.MaxInFlight = 256
	}
	if cfg.CompactEvery <= 0 {
		cfg.CompactEvery = 30 * time.Second
	}
	mux := http.NewServeMux()

	// Unhealthy means the journal can no longer be written: the server still
	// answers recalls but refuses writes, so whatever runs it should replace it.
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		stats := store.Stats()
		status := http.StatusOK
		if !stats.JournalHealthy {
			status = http.StatusServiceUnavailable
		}
		body := struct {
			memory.Stats
			Events any `json:"events,omitempty"`
		}{Stats: stats}
		if cfg.EventCounters != nil {
			body.Events = cfg.EventCounters()
		}
		writeJSON(w, status, body)
	})

	mux.HandleFunc("POST /v1/memories", func(w http.ResponseWriter, r *http.Request) {
		var req rememberRequest
		if !decode(w, r, &req) {
			return
		}
		if !namespaceAllowed(r, req.Namespace) {
			forbidNamespace(w)
			return
		}
		importance := 0.5
		if req.Importance != nil {
			importance = *req.Importance
		}
		m, err := store.Remember(memory.Memory{
			Namespace: req.Namespace, Text: req.Text, Embedding: req.Embedding,
			Importance: importance, CreatedAt: req.CreatedAt, Metadata: req.Metadata,
		})
		if err != nil {
			writeError(w, log, err)
			return
		}
		m.Embedding = nil
		writeJSON(w, http.StatusCreated, m)
	})

	mux.HandleFunc("DELETE /v1/memories/{namespace}/{id}", func(w http.ResponseWriter, r *http.Request) {
		if !namespaceAllowed(r, r.PathValue("namespace")) {
			forbidNamespace(w)
			return
		}
		if err := store.Forget(r.PathValue("namespace"), r.PathValue("id")); err != nil {
			writeError(w, log, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("POST /v1/recall", func(w http.ResponseWriter, r *http.Request) {
		var req recallRequest
		if !decode(w, r, &req) {
			return
		}
		if !namespaceAllowed(r, req.Namespace) {
			forbidNamespace(w)
			return
		}
		if req.K == 0 {
			req.K = defaultK
		}
		out, err := store.Recall(memory.Query{
			Namespace: req.Namespace, Text: req.Query, Embedding: req.Embedding,
			K: req.K, AsOf: req.AsOf, Explore: req.Explore,
		})
		if err != nil {
			writeError(w, log, err)
			return
		}
		writeJSON(w, http.StatusOK, out)
	})

	mux.HandleFunc("POST /v1/feedback", func(w http.ResponseWriter, r *http.Request) {
		var req feedbackRequest
		if !decode(w, r, &req) {
			return
		}
		if req.Reward == nil {
			http.Error(w, "reward is required", http.StatusBadRequest)
			return
		}
		if len(req.Useful) > memory.MaxK {
			http.Error(w, "too many useful ids", http.StatusBadRequest)
			return
		}
		if err := store.Feedback(req.RecallID, *req.Reward, req.Useful); err != nil {
			writeError(w, log, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	root := http.NewServeMux()
	root.Handle("/", authorizeData(cfg.Token, scopeSets(cfg.Scopes), mux))
	if cfg.AdminToken != "" {
		admin := http.NewServeMux()
		var lastCompact atomic.Int64
		admin.HandleFunc("POST /v1/compact", func(w http.ResponseWriter, _ *http.Request) {
			now, last := time.Now().UnixNano(), lastCompact.Load()
			if wait := cfg.CompactEvery - time.Duration(now-last); last != 0 && wait > 0 {
				w.Header().Set("Retry-After", strconv.Itoa(int(wait.Seconds())+1))
				http.Error(w, "compaction was requested moments ago", http.StatusTooManyRequests)
				return
			}
			if !lastCompact.CompareAndSwap(last, now) {
				http.Error(w, "compaction already requested", http.StatusTooManyRequests)
				return
			}
			if err := store.Compact(); err != nil {
				writeError(w, log, err)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		})
		admin.HandleFunc("PUT /v1/policy", func(w http.ResponseWriter, r *http.Request) {
			var req policyRequest
			if !decode(w, r, &req) {
				return
			}
			// A fixed-size array would silently drop or zero-fill a list of the
			// wrong length, so the length is checked on a slice first.
			if len(req.Weights) != policy.NumFeatures {
				http.Error(w, fmt.Sprintf("weights must have exactly %d values", policy.NumFeatures), http.StatusBadRequest)
				return
			}
			if err := store.SetWeights(policy.Features(req.Weights)); err != nil {
				writeError(w, log, err)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		})
		protected := authorize(cfg.AdminToken, admin)
		root.Handle("POST /v1/compact", protected)
		root.Handle("PUT /v1/policy", protected)
	}

	return recoverPanics(log, limitInFlight(cfg.MaxInFlight, root))
}

// limitInFlight sheds load instead of letting requests pile up without
// bound. Health checks always get through.
func limitInFlight(n int, next http.Handler) http.Handler {
	slots := make(chan struct{}, n)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			next.ServeHTTP(w, r)
			return
		}
		select {
		case slots <- struct{}{}:
			defer func() { <-slots }()
			next.ServeHTTP(w, r)
		default:
			w.Header().Set("Retry-After", "1")
			http.Error(w, "server is at capacity", http.StatusServiceUnavailable)
		}
	})
}
func authorize(token string, next http.Handler) http.Handler {
	if token == "" {
		return next
	}
	want := []byte("Bearer " + token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := []byte(r.Header.Get("Authorization"))
		if r.URL.Path != "/healthz" && subtle.ConstantTimeCompare(got, want) != 1 {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// ctxKey is unexported so nothing outside this package can set the values it
// marks in a request context.
type ctxKey int

// allowedNamespacesKey marks the set of namespaces the caller's token may
// touch. It is absent when the token is unrestricted, which the handlers read
// as "every namespace is allowed".
const allowedNamespacesKey ctxKey = iota

// scopeSets turns the configured token->namespaces lists into token->set
// lookups. A list containing "*" becomes a nil set, meaning the token is
// authenticated but not restricted to any namespace.
func scopeSets(scopes map[string][]string) map[string]map[string]bool {
	if len(scopes) == 0 {
		return nil
	}
	out := make(map[string]map[string]bool, len(scopes))
	for tok, names := range scopes {
		set := make(map[string]bool, len(names))
		for _, n := range names {
			if n == "*" {
				set = nil
				break
			}
			set[n] = true
		}
		out[tok] = set
	}
	return out
}

// authorizeData guards the data endpoints. A request authenticates with the
// open-access Token (which grants every namespace) or with any token in
// scopes. A scoped token is recorded in the request context so the handlers
// can refuse namespaces outside its set. With no Token and no scopes the
// endpoints stay open, exactly as before.
func authorizeData(token string, scopes map[string]map[string]bool, next http.Handler) http.Handler {
	if token == "" && len(scopes) == 0 {
		return next
	}
	want := []byte("Bearer " + token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			next.ServeHTTP(w, r)
			return
		}
		got := []byte(r.Header.Get("Authorization"))
		authed := token != "" && subtle.ConstantTimeCompare(got, want) == 1
		var restrictTo map[string]bool
		restricted := false
		// Every candidate is compared so the time taken does not reveal which
		// token, if any, matched.
		for tok, set := range scopes {
			if subtle.ConstantTimeCompare(got, []byte("Bearer "+tok)) == 1 {
				authed = true
				if set != nil {
					restrictTo, restricted = set, true
				}
			}
		}
		if !authed {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if restricted {
			r = r.WithContext(context.WithValue(r.Context(), allowedNamespacesKey, restrictTo))
		}
		next.ServeHTTP(w, r)
	})
}

// namespaceAllowed reports whether the caller's token may touch ns. A request
// with no recorded restriction (an unrestricted token, or no scoping at all)
// is allowed every namespace.
func namespaceAllowed(r *http.Request, ns string) bool {
	allowed, ok := r.Context().Value(allowedNamespacesKey).(map[string]bool)
	if !ok {
		return true
	}
	return allowed[ns]
}

func forbidNamespace(w http.ResponseWriter) {
	http.Error(w, "namespace not allowed for this token", http.StatusForbidden)
}

// ParseScopes reads the LETHE_TOKEN_SCOPES value: semicolon-separated entries,
// each "token:ns1,ns2". A namespace of "*" grants every namespace. An empty
// string means no scoping. Example: "alice-token:alice;ops-token:*".
func ParseScopes(s string) (map[string][]string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	out := map[string][]string{}
	for _, entry := range strings.Split(s, ";") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		tok, rest, ok := strings.Cut(entry, ":")
		tok = strings.TrimSpace(tok)
		if !ok || tok == "" {
			return nil, fmt.Errorf("token scope %q must be token:namespaces", entry)
		}
		var names []string
		for _, n := range strings.Split(rest, ",") {
			if n = strings.TrimSpace(n); n != "" {
				names = append(names, n)
			}
		}
		if len(names) == 0 {
			return nil, fmt.Errorf("token scope %q lists no namespaces", entry)
		}
		out[tok] = names
	}
	return out, nil
}

// recoverPanics keeps one failing request from taking the process down.
func recoverPanics(log *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if v := recover(); v != nil {
				log.Error("panic in handler", "path", r.URL.Path, "panic", v)
				http.Error(w, "internal error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func decode(w http.ResponseWriter, r *http.Request, into any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	dec.DisallowUnknownFields()
	if err := dec.Decode(into); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return false
	}
	return true
}

func writeError(w http.ResponseWriter, log *slog.Logger, err error) {
	switch {
	case errors.Is(err, memory.ErrInvalid):
		http.Error(w, err.Error(), http.StatusBadRequest)
	case errors.Is(err, memory.ErrNotFound):
		http.Error(w, "not found", http.StatusNotFound)
	case errors.Is(err, memory.ErrClosed):
		http.Error(w, "server is shutting down", http.StatusServiceUnavailable)
	case errors.Is(err, memory.ErrFull):
		http.Error(w, "store is full", http.StatusInsufficientStorage)
	default:
		log.Error("request failed", "error", err)
		http.Error(w, "internal error", http.StatusInternalServerError)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
