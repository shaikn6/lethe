// Command lethe-stress loads a running lethed and reports throughput and
// latency. It prints how many writes the server acknowledged, so a crash
// test can check that none of them were lost.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	base := flag.String("url", "http://127.0.0.1:8787", "server address")
	workers := flag.Int("workers", 32, "concurrent clients")
	duration := flag.Duration("duration", 10*time.Second, "how long to run")
	namespaces := flag.Int("namespaces", 8, "namespaces to spread load over")
	dim := flag.Int("dim", 64, "embedding size")
	flag.Parse()

	client := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{MaxIdleConnsPerHost: *workers}}
	post := func(path string, body any, into any) (int, error) {
		data, _ := json.Marshal(body)
		resp, err := client.Post(*base+path, "application/json", bytes.NewReader(data))
		if err != nil {
			return 0, err
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		if into != nil && resp.StatusCode < 300 {
			_ = json.Unmarshal(raw, into)
		}
		return resp.StatusCode, nil
	}

	var acked, recalls, failed atomic.Int64
	latencies := make([][]time.Duration, *workers)
	deadline := time.Now().Add(*duration)
	var wg sync.WaitGroup
	for w := range *workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ns := fmt.Sprintf("load-%d", w%*namespaces)
			vec := make([]float32, *dim)
			for i := 0; time.Now().Before(deadline); i++ {
				for j := range vec {
					vec[j] = float32((w*31+i*17+j*7)%97) / 97
				}
				start := time.Now()
				code, err := post("/v1/memories", map[string]any{
					"namespace": ns, "embedding": vec,
					"text": fmt.Sprintf("worker %d step %d: the deploy of service %d finished with status %d", w, i, i%50, i%7),
				}, nil)
				if err != nil || code != http.StatusCreated {
					failed.Add(1)
					continue
				}
				acked.Add(1)

				var out struct {
					RecallID string `json:"recall_id"`
				}
				code, err = post("/v1/recall", map[string]any{
					"namespace": ns, "query": fmt.Sprintf("status of the latest deploy of service %d", i%50),
					"embedding": vec, "k": 5, "explore": true,
				}, &out)
				if err != nil || code != http.StatusOK {
					failed.Add(1)
					continue
				}
				recalls.Add(1)
				if code, err = post("/v1/feedback", map[string]any{"recall_id": out.RecallID, "reward": float64(i % 2)}, nil); err != nil || code != http.StatusNoContent {
					failed.Add(1)
				}
				latencies[w] = append(latencies[w], time.Since(start))
			}
		}()
	}
	wg.Wait()

	var all []time.Duration
	for _, l := range latencies {
		all = append(all, l...)
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })
	pct := func(p float64) time.Duration {
		if len(all) == 0 {
			return 0
		}
		return all[int(float64(len(all)-1)*p)]
	}
	result := map[string]any{
		"acknowledged_writes": acked.Load(), "recalls": recalls.Load(), "failed_requests": failed.Load(),
		"requests_per_second": float64(acked.Load()+2*recalls.Load()) / duration.Seconds(),
		"cycle_p50_ms":        float64(pct(0.50).Microseconds()) / 1000, "cycle_p99_ms": float64(pct(0.99).Microseconds()) / 1000,
	}
	json.NewEncoder(os.Stdout).Encode(result)
}
