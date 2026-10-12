package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/shaikn6/lethe/internal/memory"
	"github.com/shaikn6/lethe/internal/server"
)

func bridge(t *testing.T) *api {
	t.Helper()
	store, err := memory.Open(memory.Options{Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(server.New(store, server.Config{Token: "tok"}, slog.New(slog.NewTextHandler(io.Discard, nil))))
	t.Cleanup(srv.Close)
	return &api{base: srv.URL, namespace: "agent", token: "tok", http: srv.Client()}
}

// exchange sends request lines and returns the decoded responses.
func exchange(t *testing.T, a *api, lines ...string) []map[string]any {
	t.Helper()
	var out bytes.Buffer
	if err := serve(strings.NewReader(strings.Join(lines, "\n")+"\n"), &out, a); err != nil {
		t.Fatal(err)
	}
	var responses []map[string]any
	dec := json.NewDecoder(&out)
	for dec.More() {
		var r map[string]any
		if err := dec.Decode(&r); err != nil {
			t.Fatal(err)
		}
		responses = append(responses, r)
	}
	return responses
}

func text(t *testing.T, resp map[string]any) (string, bool) {
	t.Helper()
	result, ok := resp["result"].(map[string]any)
	if !ok {
		t.Fatalf("no result in %v", resp)
	}
	content := result["content"].([]any)[0].(map[string]any)
	isError, _ := result["isError"].(bool)
	return content["text"].(string), isError
}

func call(id int, name, args string) string {
	return `{"jsonrpc":"2.0","id":` + string(rune('0'+id)) + `,"method":"tools/call","params":{"name":"` + name + `","arguments":` + args + `}}`
}

func TestHandshakeAndToolList(t *testing.T) {
	got := exchange(t, bridge(t),
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{}}}`,
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/list"}`,
		`{"jsonrpc":"2.0","id":3,"method":"ping"}`,
		`{"jsonrpc":"2.0","id":4,"method":"resources/list"}`,
		`not json`,
	)
	if len(got) != 5 {
		t.Fatalf("got %d responses, want 5 (the notification gets none): %v", len(got), got)
	}
	if v := got[0]["result"].(map[string]any)["protocolVersion"]; v != "2025-03-26" {
		t.Fatalf("protocolVersion = %v", v)
	}
	var names []string
	for _, tl := range got[1]["result"].(map[string]any)["tools"].([]any) {
		names = append(names, tl.(map[string]any)["name"].(string))
	}
	if strings.Join(names, ",") != "remember,recall,feedback,forget" {
		t.Fatalf("tools = %v", names)
	}
	if code := got[3]["error"].(map[string]any)["code"].(float64); code != -32601 {
		t.Fatalf("unknown method code = %v", code)
	}
	if code := got[4]["error"].(map[string]any)["code"].(float64); code != -32700 {
		t.Fatalf("parse error code = %v", code)
	}
}

func TestToolsRoundTripThroughTheRealServer(t *testing.T) {
	a := bridge(t)
	remembered, _ := text(t, exchange(t, a, call(1, "remember", `{"text":"The staging database is called heron","importance":0.9}`))[0])
	id := strings.TrimPrefix(remembered, "Remembered as ")
	if id == remembered || id == "" {
		t.Fatalf("remember said %q", remembered)
	}

	recalled, isError := text(t, exchange(t, a, call(2, "recall", `{"query":"what is the staging database called","k":3}`))[0])
	if isError || !strings.Contains(recalled, "heron") || !strings.Contains(recalled, id) {
		t.Fatalf("recall said %q", recalled)
	}
	recallID := regexp.MustCompile(`recall_id: (\w+)`).FindStringSubmatch(recalled)[1]

	if said, isError := text(t, exchange(t, a, call(3, "feedback", `{"recall_id":"`+recallID+`","reward":1,"useful":["`+id+`"]}`))[0]); isError {
		t.Fatalf("feedback failed: %s", said)
	}
	if said, isError := text(t, exchange(t, a, call(4, "feedback", `{"recall_id":"`+recallID+`","reward":1}`))[0]); !isError || !strings.Contains(said, "404") {
		t.Fatalf("second feedback should be a readable tool error, got %q (isError=%v)", said, isError)
	}
	if said, isError := text(t, exchange(t, a, call(5, "forget", `{"memory_id":"`+id+`"}`))[0]); isError {
		t.Fatalf("forget failed: %s", said)
	}
	if said, _ := text(t, exchange(t, a, call(6, "recall", `{"query":"staging database"}`))[0]); !strings.HasPrefix(said, "No memories found") {
		t.Fatalf("recall after forget said %q", said)
	}
}

func TestBadToolCallsAreReportedToTheModel(t *testing.T) {
	a := bridge(t)
	for name, line := range map[string]string{
		"unknown tool":     call(1, "explode", `{}`),
		"unknown argument": call(2, "remember", `{"text":"x","ttl":5}`),
		"empty text":       call(3, "remember", `{"text":""}`),
	} {
		if said, isError := text(t, exchange(t, a, line)[0]); !isError {
			t.Errorf("%s: expected a tool error, got %q", name, said)
		}
	}
	a.base = "http://127.0.0.1:1"
	a.http = &http.Client{}
	if said, isError := text(t, exchange(t, a, call(4, "recall", `{"query":"x"}`))[0]); !isError || !strings.Contains(said, "unreachable") {
		t.Fatalf("server down: got %q (isError=%v)", said, isError)
	}
}
