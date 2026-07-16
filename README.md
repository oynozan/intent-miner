# Intent Miner

Describe a solution. Get ranked links to people describing the pain it solves.

```
POST /runs {"input_text": "...", "icp": "..."}  ->  {"run_id": "..."}
GET  /runs/{id}                                 ->  status + stats
GET  /runs/{id}/tree                            ->  the pain tree
GET  /runs/{id}/results?node_id=&min_score=     ->  ranked links, grouped by leaf
GET  /runs/{id}/urls?node_id=&min_score=        ->  {status, urls[]} — the flat list
```

**Interactive API docs** (FastAPI, generated from the code): **Swagger UI** at
[localhost:8001/docs](http://localhost:8001/docs), **ReDoc** at
[localhost:8001/redoc](http://localhost:8001/redoc), raw **OpenAPI 3.1** spec at
`/openapi.json`. The Swagger page documents the run lifecycle, every field, the
`stats` keys, and what makes a post a lead vs. a gated-out ad — and lets you fire a
`POST /runs` from the browser.

## The one idea

**Do not search for your solution. Search for the pain.**

Nobody writes "I need a video background removal API." They write "how do I cut myself
out of a video without a green screen." Every stage exists to translate product
language into complaint language and back. `llm/prompts/expand.md` is where that
happens, and it is the product — nothing downstream can recover from getting it wrong,
because every later stage only filters what it hands them.

## Run it

```bash
cp .env.example .env      # then fill in the three keys below
docker compose up -d
curl localhost:8001/health
```

**Three surfaces, each with a primary and an optional fallback. No offline mode.**

| Surface | Primary | Fallback | Flip with |
|---|---|---|---|
| **LLM** (expand + score) | `OPENAI_API_KEY` — gpt-5-nano | `ANTHROPIC_API_KEY` — Opus/Haiku | `LLM_PROVIDER=anthropic` |
| **Embeddings** (prefilter) | `VOYAGE_API_KEY` — voyage-4-lite | `OPENAI_API_KEY` — text-embedding-3-small | `EMBED_PROVIDER=openai` |
| **Discovery** (SERP) | `SERPER_API_KEY` — Serper | `SERPAPI_API_KEY` — SerpApi | `DISCOVERY_PROVIDER=serpapi` |

Each surface uses the primary and transparently falls through to the fallback if the
primary errors or has no key. Minimum to run anything real: `OPENAI_API_KEY` (LLM) +
`VOYAGE_API_KEY` or `OPENAI_API_KEY` (embeddings) + `SERPER_API_KEY` or `SERPAPI_API_KEY`
(discovery). Voyage's first 200M embedding tokens are free (~200 runs).

gpt-5-nano is a reasoning model — its wiring (`max_completion_tokens`,
`reasoning_effort`, no `temperature`, strict `json_schema`) lives in `llm/providers.py`.
Both embedding models emit 1024-dim vectors (Voyage's `output_dimension`, OpenAI's
`dimensions`), so the `vector(1024)` schema fits either.

> **Embedding caveat.** A run's leaf vectors and candidate vectors must come from the
> *same* provider to be comparable. Both stages pick the first configured provider, so
> they stay consistent — unless the primary succeeds for one stage and fails for the
> other mid-run (e.g. an intermittent Voyage rate-limit), which would mix vector spaces
> and produce a garbage prefilter. Rare, but if you see nonsense prefilter results with
> both embedding keys set, that's the suspect.

Without a working provider a run reaches `status: expanding` and stops on
`no LLM provider configured` (or the provider's own auth error). Everything up to that
point is verified working.

Host ports are offset (API **8001**, Postgres 5432, Redis **6380**, MinIO **9002/9003**)
so this stack can run alongside the sibling `bg-remover` stack, which binds 6379 and 9000.

```bash
docker compose exec -T postgres psql -U intent -d intent_miner -c "\dt"
pytest    # 58 tests; needs redis on 6380
```

## What the research changed

The v1 spec was tested against live services before implementation. Three of its
load-bearing assumptions were wrong.

### There is no heavy fetch lane

The spec routed Quora and LinkedIn through Playwright + residential proxies. Measured
from one IP, back to back:

| | httpx (OpenSSL TLS) | curl_cffi (Chrome TLS) |
|---|---|---|
| **Quora** | 0/6 → 403 `cf-mitigated: challenge` | **3/3 → 200**, full 220–260KB payload |
| **LinkedIn** | 200, `articleBody` present | 200, `articleBody` present |

Quora's gate is **TLS fingerprint, not IP reputation** — httpx and requests delegate
the handshake to OpenSSL and emit a JA3/JA4 shared by millions of bots. Rotating IPs
would not have fixed it; a residential proxy was never the answer. LinkedIn needs
neither. **The browser stack, the proxy pool, and their entire cost line are deleted.**

The platforms take different clients on purpose. Don't unify them.

### Quora is not server-rendered

The spec said the content "sits in the DOM behind the modal — read the DOM before it
matters." There is no DOM to read. The page ships an empty root div; content arrives
**triple-JSON-encoded** in inline `.push()` calls. Miss the third decode level and
every "answer" is a JSON blob that embeds and scores as gibberish while looking
populated. Hence a checked-in 256KB live fixture and `tests/test_quora_parser.py`.

**A logged-out page carries 4 answers of 910** — and returns HTTP 200 while doing it.
Fetch success does not mean you got the page. Two consequences, both implemented:

- **Embed the question, not the body.** The question is 35 chars; the answers are
  ~13,800. Embedding the body gives a vector that is ~99.7% *other people's answers*,
  matching leaves against solutions instead of against pain.
- **`answer_count` is the saturation signal.** 910 answers = no room to be useful.
  That is the "thinly answered" half of the spec's `actionable`, free from the same
  `og:description` we already parse.

### LinkedIn is an httpx GET

`/posts/{slug}` is a deliberate SEO surface for Googlebot. The adjacent routes prove
the split: `/feed/update/` 307s to signup, `/company/x/posts/` 302s to login. Only
`/posts/` is open — so `core/urls.py` **refuses** to fetch anything else rather than
fetch a URL that redirects and scores as an empty post.

Self-correcting property: a post is only anonymously readable if its author enabled
public visibility, and non-public posts never reach the `/posts/` SEO surface. Anything
SERP-discoverable is public by construction. Non-public posts cost recall silently at
*discovery*; they are not failed fetches and no retry recovers them.

### Reddit is deferred, not cheap

Unauthenticated `.json` is **dead** — 403 on every variant, verified live, while plain
HTML returns 200. Self-serve registration has been closed since the Nov 2025 Responsible
Builder Policy, and the free tier is non-commercial by explicit definition. Full
findings in the plan; no Reddit code exists here.

## Two bugs worth knowing about

**dramatiq's `Barrier` is unusable for this pipeline.** Verified against the installed
2.2.0, not inferred:

- three retry attempts by **one** party fire a **three**-party barrier (blind `DECR`,
  no party identity)
- `wait()` before `create()` returns **True** → silent early fire
- `parties=0` asserts, and `python -O` strips asserts

`core/barriers.py` replaces it with one atomic Lua `SADD`+`SCARD` keyed on party id.
The regression tests lock all three in — don't "simplify" it back to a counter.

The trap: the *obvious* fix for dead retries (re-raise but keep
`finally: close_barrier()`) causes the early-fire bug. Read `pipeline/stages.py` before
touching an actor.

**The prefilter didn't filter.** `keep_percentile` ranked pairs globally, but
top-k-per-candidate short-circuits that — each candidate's best 3 of 300 leaves sit in
its own top 1% and clear a global 85th-percentile cut regardless of quality. Measured
at 20k × 300: **60,000 pairs → 6,000 scoring calls against a budget of 45.** No error;
it would have surfaced on the invoice. Now ranks *candidates* by their best leaf match:
same input, 3,000 survivors, 325 calls.

## Layout

```
api/          FastAPI surface
core/         config, db, redis broker, barriers, rate limits, budget, url canonicalization
discovery/    providers.py (Serper->SerpApi), serper.py, serpapi.py, common.py  (Exa declined)
llm/          client.py, embeddings.py, prompts/{expand,score}.md
pipeline/     actors.py, stages.py, prefilter.py, repo.py
scrape/       quora.py, linkedin.py   (no browser.py -- there is no browser)
tests/        58 tests; fixtures/ are live captures, not mocks
```

## Next

1. **Add the three keys** and run end-to-end on Quora. Hand-check the top 20: the real
   test is whether they are people *asking*, not people *mentioning*.
2. **Calibrate the prefilter.** `prefilter_config.keep_percentile` defaults to 0.15
   uncalibrated. Label until you have **~200 positives** (not 200 rows — at an 85% kill
   that is ~30 positives and a ±10–15pp recall CI). Sweep four `input_type` arms:
   `None/None`, `query/query`, `document/document`, `query/document`. Leaf-vs-post is
   arguably symmetric (two pain descriptions), so Voyage's "never omit input_type"
   guidance may not apply — let measured recall decide. **Tune for recall**: a false
   negative is gone forever; a false positive costs one cheap scoring slot.
3. **Bake off Haiku vs Sonnet 5 on `score_batch`.** The anti-shill call is the hard
   one. `tests/fixtures/linkedin_post.html` is a real example: it opens "Looking for a
   tool that..." and closes "Contact me today" — perfect pain-phrase match, not a lead.
4. **Re-check on 2026-09-15.** Cloudflare default-blocks Training/Agent crawler
   categories on ad-serving pages. Quora serves ads and is behind Cloudflare.

**Every yield number here has a short shelf life.** Instrument yield per platform per
run and alert on trend. Do not treat a number measured once in July as a constant.
