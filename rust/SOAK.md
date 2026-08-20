# Soak harness

A long-running load generator that pounds a configurable HTTP POST endpoint
at a fixed request rate, watches process RSS and open file descriptor count,
and exits non-zero on regression. Used to satisfy PRD §11.6 (24h continuous
load).

## Smoke test (CI, 60s)

`cargo test -p speaches-plus --test soak_smoke` boots an in-process
`EchoEngine`-backed chat router and drives it at 5 rps for 60 seconds. It
asserts:

- error rate <= 0.01%
- RSS growth <= 20% (smoke is more permissive than production)

## 24h soak (production)

```bash
# Start speaches-plus in another terminal first.
cargo run --release --example soak -- \
    --duration-sec 86400 \
    --rps 100 \
    --endpoint http://127.0.0.1:8000/v1/chat/completions
```

Exits non-zero on:

- `--max-error-rate` exceeded (default 0.0001 = 0.01%)
- `--max-rss-growth` exceeded (default 0.05 = 5%)

## Flags

| flag | default | meaning |
|------|---------|---------|
| `--endpoint` | `http://127.0.0.1:8000/v1/chat/completions` | URL to POST against |
| `--rps` | 5 | requests per second (target; achieved may be lower under load) |
| `--duration-sec` | 60 | total soak duration |
| `--report-every-sec` | 30 | period for stdout progress lines |
| `--body` | echo chat request | request body to POST |
| `--max-error-rate` | 0.0001 | fail if exceeded |
| `--max-rss-growth` | 0.05 | fail if exceeded |

## Output

Progress lines every `--report-every-sec`:

```
[soak] t=30.0s reqs=150 errs=0 err_rate=0.0000% rps=5.00 rss=42MB fds=18
```

Final summary plus PASS/FAIL exit:

```
[soak] DONE elapsed=60.0s total=300 errors=0 error_rate=0.0000% rps=5.00 rss_start=42MB rss_end=44MB growth=4.76% fds_start=18 fds_end=18
[soak] PASS
```
