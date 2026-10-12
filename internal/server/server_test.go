package server

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/shaikn6/lethe/internal/memory"
)

func newHandler(t testing.TB, token string) http.Handler {
	t.Helper()
	store, err := memory.Open(memory.Options{Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	return New(store, token, slog.New(slog.NewTextHandler(io.Discard, nil)))
}

func do(h http.Handler, method, path, body string, headers ...string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestRememberRecallFeedbackForget(t *testing.T) {
	h := newHandler(t, "")
	rec := do(h, "POST", "/v1/memories", `{"namespace":"u1","text":"Sam prefers window seats","embedding":[1,0],"metadata":{"source":"chat"}}`)
	if rec.Code != http.StatusCreated {
		t.Fatalf("remember: %d %s", rec.Code, rec.Body)
	}
	var stored struct {
		ID         string  `json:"id"`
		Importance float64 `json:"importance"`
	}
	json.Unmarshal(rec.Body.Bytes(), &stored)
	if stored.ID == "" || stored.Importance != 0.5 || strings.Contains(rec.Body.String(), "embedding") {
		t.Fatalf("remember body = %s", rec.Body)
	}

	rec = do(h, "POST", "/v1/recall", `{"namespace":"u1","query":"which seat does Sam like","embedding":[1,0],"explore":true}`)
	var got struct {
		RecallID string `json:"recall_id"`
		Memories []struct {
			ID   string `json:"id"`
			Text string `json:"text"`
		} `json:"memories"`
	}
	json.Unmarshal(rec.Body.Bytes(), &got)
	if rec.Code != http.StatusOK || len(got.Memories) != 1 || got.Memories[0].ID != stored.ID {
		t.Fatalf("recall: %d %s", rec.Code, rec.Body)
	}

	if rec = do(h, "POST", "/v1/feedback", `{"recall_id":"`+got.RecallID+`","reward":1}`); rec.Code != http.StatusNoContent {
		t.Fatalf("feedback: %d %s", rec.Code, rec.Body)
	}
	if rec = do(h, "POST", "/v1/feedback", `{"recall_id":"`+got.RecallID+`","reward":1}`); rec.Code != http.StatusNotFound {
		t.Fatalf("repeated feedback: %d, want 404", rec.Code)
	}
	if rec = do(h, "GET", "/healthz", ""); !strings.Contains(rec.Body.String(), `"policy_updates":1`) || !strings.Contains(rec.Body.String(), `"memories":1`) {
		t.Fatalf("healthz = %s", rec.Body)
	}

	if rec = do(h, "DELETE", "/v1/memories/u1/"+stored.ID, ""); rec.Code != http.StatusNoContent {
		t.Fatalf("forget: %d %s", rec.Code, rec.Body)
	}
	if rec = do(h, "DELETE", "/v1/memories/u1/"+stored.ID, ""); rec.Code != http.StatusNotFound {
		t.Fatalf("second forget: %d, want 404", rec.Code)
	}
	rec = do(h, "POST", "/v1/recall", `{"namespace":"u1","query":"seat"}`)
	if !strings.Contains(rec.Body.String(), `"memories":[]`) {
		t.Fatalf("recall after forget = %s", rec.Body)
	}
}

func TestBadRequests(t *testing.T) {
	h := newHandler(t, "")
	cases := map[string]struct{ path, body string }{
		"malformed json":      {"/v1/memories", `{"namespace":`},
		"unknown field":       {"/v1/memories", `{"namespace":"n","text":"x","ttl":5}`},
		"missing namespace":   {"/v1/memories", `{"text":"x"}`},
		"empty text":          {"/v1/memories", `{"namespace":"n","text":""}`},
		"importance range":    {"/v1/memories", `{"namespace":"n","text":"x","importance":7}`},
		"k out of range":      {"/v1/recall", `{"namespace":"n","query":"x","k":5000}`},
		"negative k":          {"/v1/recall", `{"namespace":"n","query":"x","k":-2}`},
		"feedback no reward":  {"/v1/feedback", `{"recall_id":"abc"}`},
		"reward out of range": {"/v1/feedback", `{"recall_id":"abc","reward":3}`},
		"body too large":      {"/v1/memories", `{"namespace":"n","text":"` + strings.Repeat("a", maxBodyBytes) + `"}`},
	}
	for name, tc := range cases {
		if rec := do(h, "POST", tc.path, tc.body); rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400", name, rec.Code)
		}
	}
	if rec := do(h, "GET", "/v1/recall", ""); rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET /v1/recall = %d, want 405", rec.Code)
	}
}

func TestBearerToken(t *testing.T) {
	h := newHandler(t, "s3cret")
	body := `{"namespace":"n","text":"x"}`
	if rec := do(h, "POST", "/v1/memories", body); rec.Code != http.StatusUnauthorized {
		t.Fatalf("no token: %d, want 401", rec.Code)
	}
	if rec := do(h, "POST", "/v1/memories", body, "Authorization", "Bearer wrong"); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong token: %d, want 401", rec.Code)
	}
	if rec := do(h, "POST", "/v1/memories", body, "Authorization", "Bearer s3cret"); rec.Code != http.StatusCreated {
		t.Fatalf("right token: %d, want 201", rec.Code)
	}
	if rec := do(h, "GET", "/healthz", ""); rec.Code != http.StatusOK {
		t.Fatalf("healthz without token: %d, want 200", rec.Code)
	}
}

func TestPanicInHandlerBecomesA500(t *testing.T) {
	h := recoverPanics(slog.New(slog.NewTextHandler(io.Discard, nil)), http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic("boom")
	}))
	if rec := do(h, "GET", "/", ""); rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
}

// Arbitrary request bodies must never crash the server or produce a 500.
func FuzzAPI(f *testing.F) {
	h := newHandler(f, "")
	do(h, "POST", "/v1/memories", `{"namespace":"n","text":"seed memory","embedding":[1,0]}`)
	for _, seed := range []string{
		`{"namespace":"n","text":"hello","embedding":[0.5,0.5]}`,
		`{"namespace":"n","query":"hello","k":3,"explore":true}`,
		`{"recall_id":"x","reward":0.5}`,
		`{"namespace":"n","query":"","embedding":[1e308,1e308]}`,
		`{"namespace":"n","text":"x","created_at":"0001-01-01T00:00:00Z"}`,
		`[]`, `null`, `{"k":-1}`, ``,
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, body string) {
		for _, path := range []string{"/v1/memories", "/v1/recall", "/v1/feedback"} {
			if rec := do(h, "POST", path, body); rec.Code >= 500 {
				t.Fatalf("POST %s with %q returned %d: %s", path, body, rec.Code, rec.Body)
			}
		}
	})
}
