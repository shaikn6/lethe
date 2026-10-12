// Command lethe-mcp exposes a lethe server to MCP clients over stdio.
//
// It speaks the tools part of the Model Context Protocol: JSON-RPC 2.0, one
// message per line. Memories live in lethed; this process only translates.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"time"
)

const latestProtocol = "2025-06-18"

var knownProtocols = []string{"2024-11-05", "2025-03-26", latestProtocol}

type request struct {
	ID     json.RawMessage `json:"id"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

func object(required []string, properties map[string]any) map[string]any {
	return map[string]any{"type": "object", "properties": properties, "required": required, "additionalProperties": false}
}

var tools = []tool{
	{
		Name:        "remember",
		Description: "Store something worth knowing later: a fact, a preference, a decision. One idea per memory.",
		InputSchema: object([]string{"text"}, map[string]any{
			"text":       map[string]any{"type": "string", "description": "What to remember."},
			"importance": map[string]any{"type": "number", "minimum": 0, "maximum": 1, "description": "How much this matters, 0 to 1."},
		}),
	},
	{
		Name:        "recall",
		Description: "Find stored memories that help with a question. Returns a recall_id; pass it to feedback afterwards so recall improves.",
		InputSchema: object([]string{"query"}, map[string]any{
			"query": map[string]any{"type": "string", "description": "What you want to know."},
			"k":     map[string]any{"type": "integer", "minimum": 1, "maximum": 20, "description": "How many memories to return (default 5)."},
		}),
	},
	{
		Name:        "feedback",
		Description: "Report whether a recall helped. This is how the memory learns which memories to return.",
		InputSchema: object([]string{"recall_id", "reward"}, map[string]any{
			"recall_id": map[string]any{"type": "string"},
			"reward":    map[string]any{"type": "number", "minimum": 0, "maximum": 1, "description": "1 if the memories answered the question, 0 if they were useless."},
			"useful":    map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Ids of the returned memories you actually used."},
		}),
	},
	{
		Name:        "forget",
		Description: "Delete a memory permanently.",
		InputSchema: object([]string{"memory_id"}, map[string]any{
			"memory_id": map[string]any{"type": "string"},
		}),
	},
}

// api is the small part of the lethe HTTP API this bridge needs.
type api struct {
	base, namespace, token string
	http                   *http.Client
}

func (a *api) do(method, path string, body any) ([]byte, error) {
	var payload io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		payload = bytes.NewReader(data)
	}
	req, err := http.NewRequest(method, a.base+path, payload)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if a.token != "" {
		req.Header.Set("Authorization", "Bearer "+a.token)
	}
	resp, err := a.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("memory server unreachable: %w", err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("memory server said %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return data, nil
}

// call runs one tool and returns the text shown to the model.
func (a *api) call(name string, args json.RawMessage) (string, error) {
	dec := json.NewDecoder(bytes.NewReader(args))
	dec.DisallowUnknownFields()
	switch name {
	case "remember":
		var in struct {
			Text       string   `json:"text"`
			Importance *float64 `json:"importance"`
		}
		if err := dec.Decode(&in); err != nil {
			return "", err
		}
		body := map[string]any{"namespace": a.namespace, "text": in.Text}
		if in.Importance != nil {
			body["importance"] = *in.Importance
		}
		data, err := a.do(http.MethodPost, "/v1/memories", body)
		if err != nil {
			return "", err
		}
		var out struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(data, &out); err != nil {
			return "", err
		}
		return "Remembered as " + out.ID, nil
	case "recall":
		var in struct {
			Query string `json:"query"`
			K     int    `json:"k"`
		}
		if err := dec.Decode(&in); err != nil {
			return "", err
		}
		if in.K == 0 {
			in.K = 5
		}
		data, err := a.do(http.MethodPost, "/v1/recall", map[string]any{"namespace": a.namespace, "query": in.Query, "k": in.K, "explore": true})
		if err != nil {
			return "", err
		}
		var out struct {
			RecallID string `json:"recall_id"`
			Memories []struct {
				ID        string    `json:"id"`
				Text      string    `json:"text"`
				CreatedAt time.Time `json:"created_at"`
			} `json:"memories"`
		}
		if err := json.Unmarshal(data, &out); err != nil {
			return "", err
		}
		if len(out.Memories) == 0 {
			return "No memories found. recall_id: " + out.RecallID, nil
		}
		var b strings.Builder
		fmt.Fprintf(&b, "recall_id: %s\n", out.RecallID)
		for _, m := range out.Memories {
			fmt.Fprintf(&b, "- [%s] (%s) %s\n", m.ID, m.CreatedAt.Format("2006-01-02"), m.Text)
		}
		b.WriteString("Afterwards, call feedback with this recall_id and the ids you used.")
		return b.String(), nil
	case "feedback":
		var in struct {
			RecallID string   `json:"recall_id"`
			Reward   float64  `json:"reward"`
			Useful   []string `json:"useful"`
		}
		if err := dec.Decode(&in); err != nil {
			return "", err
		}
		body := map[string]any{"recall_id": in.RecallID, "reward": in.Reward}
		if in.Useful != nil {
			body["useful"] = in.Useful
		}
		if _, err := a.do(http.MethodPost, "/v1/feedback", body); err != nil {
			return "", err
		}
		return "Feedback recorded.", nil
	case "forget":
		var in struct {
			MemoryID string `json:"memory_id"`
		}
		if err := dec.Decode(&in); err != nil {
			return "", err
		}
		path := "/v1/memories/" + url.PathEscape(a.namespace) + "/" + url.PathEscape(in.MemoryID)
		if _, err := a.do(http.MethodDelete, path, nil); err != nil {
			return "", err
		}
		return "Forgotten.", nil
	}
	return "", fmt.Errorf("unknown tool %q", name)
}

// serve answers MCP requests from in on out until in is closed.
func serve(in io.Reader, out io.Writer, a *api) error {
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 64<<10), 4<<20)
	enc := json.NewEncoder(out)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var req request
		if err := json.Unmarshal(line, &req); err != nil {
			if err := enc.Encode(response{JSONRPC: "2.0", ID: json.RawMessage("null"), Error: &rpcError{Code: -32700, Message: "parse error"}}); err != nil {
				return err
			}
			continue
		}
		if req.ID == nil {
			continue // a notification; nothing to answer
		}
		resp := response{JSONRPC: "2.0", ID: req.ID}
		switch req.Method {
		case "initialize":
			var params struct {
				ProtocolVersion string `json:"protocolVersion"`
			}
			_ = json.Unmarshal(req.Params, &params)
			version := latestProtocol
			if slices.Contains(knownProtocols, params.ProtocolVersion) {
				version = params.ProtocolVersion
			}
			resp.Result = map[string]any{
				"protocolVersion": version,
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "lethe", "version": "0.1.0"},
				"instructions":    "Long-term memory. Recall before answering questions that depend on earlier context, remember new facts, and always send feedback after a recall.",
			}
		case "ping":
			resp.Result = map[string]any{}
		case "tools/list":
			resp.Result = map[string]any{"tools": tools}
		case "tools/call":
			var params struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}
			if err := json.Unmarshal(req.Params, &params); err != nil {
				resp.Error = &rpcError{Code: -32602, Message: "invalid params"}
				break
			}
			if len(params.Arguments) == 0 {
				params.Arguments = json.RawMessage("{}")
			}
			text, err := a.call(params.Name, params.Arguments)
			if err != nil {
				// Tool failures go back as a result the model can read and react to.
				resp.Result = map[string]any{"isError": true, "content": []map[string]string{{"type": "text", "text": err.Error()}}}
				break
			}
			resp.Result = map[string]any{"content": []map[string]string{{"type": "text", "text": text}}}
		default:
			resp.Error = &rpcError{Code: -32601, Message: "method not found: " + req.Method}
		}
		if err := enc.Encode(resp); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func main() {
	base := flag.String("url", "http://127.0.0.1:8787", "lethe server address")
	namespace := flag.String("namespace", "default", "namespace this client reads and writes")
	flag.Parse()
	a := &api{
		base: strings.TrimRight(*base, "/"), namespace: *namespace, token: os.Getenv("LETHE_TOKEN"),
		http: &http.Client{Timeout: 15 * time.Second},
	}
	if err := serve(os.Stdin, os.Stdout, a); err != nil {
		fmt.Fprintln(os.Stderr, "lethe-mcp:", err)
		os.Exit(1)
	}
}
