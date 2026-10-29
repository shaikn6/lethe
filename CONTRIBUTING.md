# Contributing

Thanks for taking the time to contribute.

## Development setup

Go 1.26.10 or newer (the `go` line in `go.mod`).

```bash
go build ./...
go vet ./...
go test ./... -count=1 -race     # what CI runs
golangci-lint run                # CI uses golangci-lint v2.15.0
```

Python client (`python/`, Python 3.10+):

```bash
go build -o bin/lethed ./cmd/lethed      # the tests start this binary on a free port
python -m venv .venv && .venv/bin/pip install -e './python[dev]'
.venv/bin/pytest python/tests
```

The LoCoMo benchmark (`bench/locomo.py`) needs `data/locomo10.json` from the
LoCoMo repository; see the Quickstart in the README.

## Ground rules

- **One logical change per PR.** Keep diffs reviewable.
- **Tests are not optional.** New behaviour needs a test; bug fixes need a test
  that fails before the fix.
- **The build must be green:** `go build ./...`, `go vet ./...`,
  `golangci-lint run` and `go test -race ./...` all pass.
- **Never weaken the journal.** A change that acknowledges a write must keep the
  write-ahead order: journal first, then acknowledge.
- **Numbers in the README come from code at HEAD.** If a change moves a
  benchmark result, re-run it and update the table and its caveats in the same PR.

## Commit messages

Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

## Reporting bugs / proposing features

Use the issue templates. For anything security-related, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.
