// Package server is the HTTP API in front of the memory store.
package server

import (
	"crypto/subtle"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/shaikn6/lethe/internal/memory"
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

type feedbackRequest struct {
	RecallID string   `json:"recall_id"`
	Reward   *float64 `json:"reward"`
}

// New returns the API handler. When token is not empty, every /v1 request
// must carry it as a bearer token.
func New(store *memory.Store, token string, log *slog.Logger) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, store.Stats())
	})

	mux.HandleFunc("POST /v1/memories", func(w http.ResponseWriter, r *http.Request) {
		var req rememberRequest
		if !decode(w, r, &req) {
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
		if err := store.Feedback(req.RecallID, *req.Reward); err != nil {
			writeError(w, log, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	return recoverPanics(log, authorize(token, mux))
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
