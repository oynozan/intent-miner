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

Nobody writes "I need a video background removal API." They write "how do I remove the
background from a video without a green screen." Every stage exists to translate product
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

> **Embedding provider is locked per run.** A run's leaf and candidate vectors must
> come from the *same* provider or the cosine gate is meaningless. The prefilter embeds
> candidates first (the larger load, so they pick the provider — and fall back if the
> primary can't serve them), then embeds leaves forced to that same provider. The run
> records which one it used in `stats.embed_provider`.
>
> **Voyage's free tier is too small for candidate volume.** Without a payment method,
> Voyage caps at 10K tokens/min — a batch of a few hundred candidates blows past it, so
> in practice a Voyage-primary run falls back to OpenAI for embeddings (and locks leaves
> to OpenAI too, consistently). Add a Voyage payment method (the 200M free tokens still
> apply) or just set `EMBED_PROVIDER` so OpenAI is primary if you'd rather not fall back.

Without a working provider a run reaches `status: expanding` and stops on
`no LLM provider configured` (or the provider's own auth error). Everything up to that
point is verified working.

Host ports are offset (API **8001**, Postgres 5432, Redis **6380**, MinIO **9002/9003**)
so this stack can run alongside the sibling `bg-remover` stack, which binds 6379 and 9000.

```bash
docker compose exec -T postgres psql -U intent -d intent_miner -c "\dt"
pytest    # 173 tests; needs redis on 6380
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
neither. **The proxy pool and its entire cost line are deleted**, and so is the browser
*fetch lane* — no page is ever rendered to be scraped. A headless Chromium survives for
exactly one job, minting Reddit's cookie jar about once per session; see "Reddit's gate
is state, not fingerprint".

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

**`www.` is load-bearing on LinkedIn, and only on LinkedIn.** Same post, same client,
same second:

```
https://www.linkedin.com/posts/{slug}  ->  200,  335,526 bytes, articleBody present
https://linkedin.com/posts/{slug}      ->  200,   20,488 bytes, no ld+json at all
```

No redirect — the bare host just answers with a stub. `canonicalize()` stripped `www.`
for every platform, so **every** LinkedIn candidate got the stub, parsed to an empty
body, and matched `looks_throttled` (no post ld+json node) → three retries each, then
the throttle circuit breaker tripped and skipped the rest of the run. It read as IP
throttling for the obvious reason: it looks exactly like it. `canonicalize()` now pins
LinkedIn to `www.linkedin.com`; Quora and Reddit still drop `www.`.

**The ld+json `@type` was renamed.** Public posts now emit `DiscussionForumPosting`,
not `SocialMediaPosting`. Matching only the old name is silent rather than fatal — the
body still arrives via the `og:description` fallback, so fetches look fine — but tier 3
carries no `datePublished` and no `commentCount`, so `posted_at` is `None` and
`engagement` is `0` on a page that had both. Recency decay and the engagement signal
were dead for every LinkedIn row. `scrape/linkedin.py` accepts both names.

### Reddit's gate is state, not fingerprint

**This section previously said Reddit was unfetchable. That was wrong, and the
correction is the most valuable thing in this file.** The original finding — HTML
returns 200 with a "Please wait for verification" interstitial, `old.reddit.com` and
every `.json` variant return 403, all of it holding under `curl_cffi`'s Chrome TLS
fingerprint — was real but was generalised from a single variable. Every one of those
probes was made *with no cookies*. Measured again, same Chrome TLS throughout:

```
no cookies                        403 on every .json    (0/6)
cookies minted by a real browser  200 on every .json    (20/20, 0.83s each)
```

So the browser is needed **per session, not per URL**. After a mint, the cheap fetcher
reads a few KB of JSON per post instead of a 1MB DOM. This cost the platform that
produces most of the leads its body text, its date and its engagement for as long as the
wrong conclusion stood.

**A jar is spent by volume, and that number is small.** The 20/20 above suggested "mint
once per TTL", which the first real run disproved: a jar that had worked earlier 429'd on
30 consecutive URLs, and a fresh one served the same URLs 6/6 immediately. A session is
good for roughly **30 fetches**, so a run of a few hundred candidates re-mints ten-odd
times — the *replacement* path is the one that has to work, not the cold-start path.
`jar(stale=...)` exists for exactly that, and the 2h TTL is now only an idle bound.

**Why a real browser and nothing lighter.** Both cheaper options yield a single
`edgebucket` cookie, which unlocks nothing:

```
curl_cffi GET of the homepage      1 cookie
obscura --stealth --dump cookies   1 cookie
headless chromium                 12 cookies -> works
```

The jar Reddit wants includes `loid`, `csv` and `pxrc` (that last one looks like
PerimeterX), written by JavaScript neither a bare HTTP client nor obscura's V8 executes
fully. Hence Playwright — used only here, roughly once per TTL, never per URL. It runs in
a **child process**: Playwright's sync API drives its driver over an asyncio subprocess
transport, which on Windows only works on the main thread, so calling it from a dramatiq
worker thread fails with `[Errno 9] Bad file descriptor` — observed 58 times in one run,
every re-mint silently degrading to the SERP fallback.

**Enrichment is strictly additive.** `_fetch_reddit` never raises and never records a
failure: a stale jar, a Reddit change or one bad response falls back to `_from_serp`, so
the worst outcome is exactly the old behaviour. Reddit supplies most of this pipeline's
leads; a fetch layer that can turn them into `failed` rows is the worse trade.

`.json` gives `selftext` (the real body), `created_utc` (real recency, replacing the 0.5
unknown-date default), `ups` and `num_comments`. The last two are deliberately kept
apart: **upvotes are reach** and feed `engagement`; **comments are saturation** and feed
`answers_total`, which the scorer reads as `existing_answer_count`. Summing them would
make one post simultaneously more attractive and more crowded.

Quora went the other way and is now SERP-only — see `_from_serp`, which both platforms
share. Its gate became a *quota*, not a rate: pacing bought nothing, only ~36 minutes of
near-idle restored partial service, and rotating IPs did not buy past it (0/25 datacenter,
2/15 residential, lifting to only 4/12 across 8 sessions per URL). That costs little,
because the fetch was retrieving something already held — `scrape/quora.py` embeds the
*question*, which is what the SERP title already carries; 270/270 Quora rows in the last
full run had both a title and a snippet.

Discovery is still Google-only by construction. Reddit's robots.txt has allowed only
Google since July 2024, so every non-Google index is frozen at that date: DuckDuckGo
returns 10 `site:reddit.com` results whose newest thread predates the block, 1 with
`&df=y`, and 0 for any 2025+ topic. Bing's API is retired, Brave's free tier is gone,
Google's Custom Search JSON API is closed to new customers and retires 2027-01-01. This
is why Serper and SerpApi are not two of several options.

Rows that land on `_from_serp` (all Quora, plus Reddit when the jar fails) carry no
`posted_at`, no `engagement` and no `answers_total`, so recency decay and the saturation
filter are inert for them — they score at the neutral 0.5 recency. Both vendors return a
`date` on organic results if that starts mattering.

## Three bugs worth knowing about

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

**`dramatiq.Retry` consumes the retry budget** — despite reading like control flow. Also
verified against the installed 2.2.0, in `Retries.after_process_message`:

```python
message.options["retries"] += 1                              # unconditional
...
if isinstance(exception, Retry) and exception.delay is not None:   # type checked AFTER
```

`core/limits.py` used to `raise Retry` on a missed rate-limit slot, commented as
requeueing "without consuming a retry budget". With `max_retries=3`, three missed slots
killed a message that had never made a single request. Measured on one run when the
Quora limiter was first wired up: **587 limiter requeues, 195 of 199 candidates terminal
at `{'retries': 3, 'max_retries': 3}`, 8 actual Quora responses.** The pacing worked
perfectly and the run still got nothing — the limiter had become the thing killing the
messages, and every symptom pointed at Quora.

`rate_limited` now blocks for a slot (jittered, 75s cap — sized for the widest window in
`LIMITS`, since a thread arriving at a saturated 60s window may wait most of a window for
it to roll) and only requeues if genuinely saturated. Don't "simplify" it back to a bare
raise. `tests/test_limits.py` asserts the upstream ordering, so if a future dramatiq
starts checking the type first, the test says so.

**Rate ceilings are a shape, not a number.** `LIMITS` holds a *tuple* of
`(requests, window)` pairs per domain, all held at once, because a vendor's ceiling is
not always one number. Serper was configured as 50 per 10s — the same average as its
documented 5/s — but `WindowRateLimiter` is a window limiter, not a smoother: it admits
all 50 the instant the window opens, so 2×16 discover threads hit the per-second wall
together and ~85% of queries 429'd. Quora holds a longer quota *on top of* a per-second
rate, which a short probe cannot see: 24 requests over 24s passed 100%, while ~300 over
~7min bled 109 429s. Widen only against a measurement.

**The prefilter didn't filter.** `keep_percentile` ranked pairs globally, but
top-k-per-candidate short-circuits that — each candidate's best 3 of 300 leaves sit in
its own top 1% and clear a global 85th-percentile cut regardless of quality. Measured
at 20k × 300: **60,000 pairs → 6,000 scoring calls against a budget of 45.** No error;
it would have surfaced on the invoice. Now ranks *candidates* by their best leaf match:
same input, 3,000 survivors, 325 calls.

## Selling it on OKX.AI

Two services, priced per call in USDT and settled over x402. `/runs` stays free and
unchanged — it is the human API and its shape is not a promise. `/a2mcp/*` is the paid
contract, and its price and URL are written on-chain at registration, so it gets its own
routes that an internal refactor cannot quietly reshape.

| Service | Endpoint | Method | Price |
|---|---|---|---|
| Create a job from a keyword | `/a2mcp/jobs` | POST | **0.05 USDT** |
| Job status + discovered links | `/a2mcp/jobs/status?job_id=…` | GET | **0.001 USDT** |

A call with no `PAYMENT-SIGNATURE` returns **402** with the challenge in
`PAYMENT-REQUIRED` (base64) *and* in the body; pay it, replay, and the response carries a
`PAYMENT-RESPONSE` receipt. Status is 50× cheaper than create on purpose — a buyer
polling a job they already paid for should not be charged like they are starting another.

**The status service takes `job_id` as a query param, not a path segment.** A listing
stores one fixed URL, so `/a2mcp/jobs/{id}` could not be registered; the buyer-side
`payment quote <url> --param job_id=…` puts known params in the query string for GET.
Same reason the create service declares `"method": "POST"` in its `outputSchema` — the
buyer probes with GET by default, and an undeclared POST endpoint answers 405 and reads
as unreachable rather than as priced.

**One integration is still open: the facilitator.** Verifying that a presented signature
is real, funded and unspent needs the service that holds the on-chain view, and the
`onchainos payment` CLI is buyer-side only (pay / quote / decode-receipt / pay-local /
a2a-pay / charge / session / subscription — no `verify`). `core/x402.py` implements the
public x402 facilitator contract (`POST /verify`, `POST /settle`, with `paymentPayload` +
`paymentRequirements`); point `X402_FACILITATOR_VERIFY_URL` / `_SETTLE_URL` at OKX's
facilitator and confirm those field names before taking real money.

Until then the paid surface **fails closed** — a presented payment it cannot verify gets
a 500, never the work. There is deliberately no bypass flag: the alternative failure mode
is a service that hands out paid work for free and looks perfectly healthy doing it. Note
the asymmetry in `core/x402.py` — a *malformed or rejected* payment is a 402 ("pay me"),
but *our* inability to verify is a 500, because telling a buyer who already paid to pay
again is the wrong side to fail on.

### The listing itself

Registering is an ASP identity (one per wallet, on XLayer only, gas paid by OKX) plus two
services. Both service descriptions are three numbered parts because both are ordinary
per-call services — the two-part form is only for subscription-priced ones, and A2MCP has
no subscription. Prices below must stay equal to `A2MCP_PRICE_*` in `.env`; the listing is
on-chain and the endpoint is not, so they drift silently.

```
Agent name    Intent Miner
Description   Finds people publicly describing a problem your product solves, and
              ranks them by how likely they are to be a real lead.
Avatar        REQUIRED — an image file, 1:1, <1MB. Links are rejected.

Service 1     Keyword Pain Discovery Job   A2MCP   0.05    POST /a2mcp/jobs
  1. Turns one keyword into a tree of specific user pains, then searches Reddit, Quora
     and LinkedIn for people describing them. For founders and marketers validating demand.
  2. A keyword describing your product or solution, and optionally a one-line
     ideal-customer hint to bias the ranking.
  3. Returns a job id immediately; mining runs asynchronously. Delivered as JSON, read
     through the job status service. No copy-trading.

Service 2     Job Status And Lead Links    A2MCP   0.001   GET /a2mcp/jobs/status
  1. Returns the current stage of a mining job and the ranked links found so far, best
     match first. For buyers polling a job they already created.
  2. The job id returned when the job was created, and optionally a minimum score to
     filter out vendor ads and dead threads.
  3. Delivered as JSON: job status plus a de-duplicated list of result URLs. Callable
     while the job is still running. No copy-trading.
```

Descriptions carry no links, no tech-stack names, no disclaimers and no outcome
promises — all four are rejected by listing QA, and the last one semantically rather than
by keyword.

## Layout

```
api/          FastAPI surface: routes.py (free /runs), a2mcp.py (paid, x402-priced)
core/         config, db, redis broker, barriers, rate limits, budget, url canonicalization,
              x402.py (seller side: mint the 402 challenge, verify the paid replay)
discovery/    providers.py (Serper->SerpApi), serper.py, serpapi.py, common.py  (Exa declined)
llm/          client.py, embeddings.py, prompts/{expand,score}.md
pipeline/     actors.py, stages.py, prefilter.py, repo.py
scrape/       quora.py (parser kept; fetch retired -- SERP-only), linkedin.py,
              reddit.py (cookie-jar mint + .json fetch; the only browser in the stack)
tests/        173 tests; fixtures/ are live captures, not mocks
```

## Next

1. **Add the three keys** and run end-to-end on Reddit — it carries the fetch path and
   most of the leads (Quora is SERP-only now). Hand-check the top 20: the real test is
   whether they are people *asking*, not people *mentioning*.
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
