// Command lethed runs the memory server.
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/shaikn6/lethe/internal/memory"
	"github.com/shaikn6/lethe/internal/server"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8787", "listen address")
	journal := flag.String("journal", "lethe.journal", "journal file; empty keeps everything in memory")
	fsync := flag.Bool("fsync", false, "fsync every journal write (survives power loss, slower)")
	seed := flag.Uint64("seed", 0, "exploration seed; 0 picks a random one")
	maxInFlight := flag.Int("max-inflight", 0, "requests served at once before others get 503; 0 keeps the default of 256")
	maxMemories := flag.Int("max-memories", 0, "live memories the store accepts; 0 keeps the default of one million")
	learningRate := flag.Float64("learning-rate", 0, "policy learning rate; 0 keeps the default")
	temperature := flag.Float64("temperature", 0, "exploration temperature; 0 keeps the default")
	flag.Parse()

	log := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	store, err := memory.Open(memory.Options{
		JournalPath: *journal, Fsync: *fsync, Seed: *seed, MaxMemories: *maxMemories,
		LearningRate: *learningRate, Temperature: *temperature,
	})
	if err != nil {
		log.Error("open store", "error", err)
		os.Exit(1)
	}
	stats := store.Stats()
	log.Info("store ready", "memories", stats.Memories, "policy_updates", stats.Updates, "addr", *addr)

	srv := &http.Server{
		Addr: *addr,
		Handler: server.New(store, server.Config{
			Token:       os.Getenv("LETHE_TOKEN"),
			AdminToken:  os.Getenv("LETHE_ADMIN_TOKEN"),
			MaxInFlight: *maxInFlight,
		}, log),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       90 * time.Second,
		MaxHeaderBytes:    16 << 10,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()

	if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		log.Error("server stopped", "error", err)
	}
	if err := store.Close(); err != nil {
		log.Error("close store", "error", err)
		os.Exit(1)
	}
	log.Info("shut down cleanly")
}
