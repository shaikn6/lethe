# syntax=docker/dockerfile:1
FROM golang:1.26 AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/lethed ./cmd/lethed

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/lethed /lethed
VOLUME /data
EXPOSE 8787
USER nonroot:nonroot
ENTRYPOINT ["/lethed", "-addr", "0.0.0.0:8787", "-journal", "/data/lethe.journal", "-fsync"]
