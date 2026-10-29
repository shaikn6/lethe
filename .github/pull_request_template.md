## What

<!-- One or two sentences on the change. -->

## Why

<!-- The problem this solves or the behaviour it adds. -->

## Checklist

- [ ] `go build ./...`, `go vet ./...`, `golangci-lint run` pass
- [ ] `go test ./... -count=1 -race` passes
- [ ] `pytest python/tests` passes if the Python client changed
- [ ] New behaviour has tests; bug fixes have a regression test
- [ ] README numbers re-run and updated if a benchmark result moved
