CREATE EXTENSION IF NOT EXISTS vector;

-- Dimension is 1024 (voyage-4-lite), NOT the 1536 the original spec carried.
-- 1536 is an OpenAI ada-002/3-small native size; the Voyage 4 family supports
-- 256/512/1024/2048 and has no 1536 option. 1024 over 2048 because it stays under
-- pgvector's 2000-dim HNSW ceiling, leaving the option open if candidates are ever
-- retained at scale. Changing this later is a re-embed, not a migration.

CREATE TABLE runs (
    id          uuid PRIMARY KEY,
    input_text  text NOT NULL,
    icp         text,
    status      text NOT NULL DEFAULT 'pending',
    stats       jsonb NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE TABLE nodes (
    id             uuid PRIMARY KEY,
    run_id         uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    parent_id      uuid REFERENCES nodes(id) ON DELETE CASCADE,
    depth          int NOT NULL,
    kind           text NOT NULL,
    label          text NOT NULL,
    description    text NOT NULL,
    pain_phrases   text[] NOT NULL DEFAULT '{}',
    negative_terms text[] NOT NULL DEFAULT '{}',
    icp_hint       text,
    -- Leaf embeddings persist: a few hundred vectors, reused every run, trivially
    -- small, and they enable cross-run leaf reuse.
    embedding      vector(1024)
);
CREATE INDEX ON nodes (run_id, kind);

CREATE TABLE queries (
    id       uuid PRIMARY KEY,
    run_id   uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    node_id  uuid NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    platform text NOT NULL,
    channel  text NOT NULL,
    q        text NOT NULL,
    q_hash   text NOT NULL,
    depth    int NOT NULL DEFAULT 10,   -- Serper bills 2 credits for num > 10, so depth is per-query
    status   text NOT NULL DEFAULT 'pending',
    error    text,
    hits     int NOT NULL DEFAULT 0,
    UNIQUE (run_id, channel, q_hash)
);
CREATE INDEX ON queries (run_id, status);

CREATE TABLE candidates (
    id           uuid PRIMARY KEY,
    run_id       uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    url          text NOT NULL,
    url_hash     text NOT NULL,
    platform     text NOT NULL,
    title        text,
    snippet      text,
    body         text,
    author       text,
    posted_at    timestamptz,
    engagement   int NOT NULL DEFAULT 0,
    raw_key      text,
    fetch_status text NOT NULL DEFAULT 'pending',
    fetch_error  text,
    -- Completeness oracle for Quora: logged-out pages carry only ~3.5 answers of N.
    -- answers_seen/answers_total is how we know whether we fetched the post or a
    -- third of it. A truncated fetch returns HTTP 200, so status alone can't tell us.
    answers_seen  int,
    answers_total int,
    UNIQUE (run_id, url_hash)
);
CREATE INDEX ON candidates (run_id, fetch_status);
-- The spec omitted this. Candidates accumulate across runs; every prefilter and
-- ranking query is run-scoped, and without this the "20k rows" assumption decays
-- into a full-table scan run over run.
CREATE INDEX ON candidates (run_id);

-- Candidate embeddings live off the hot row on purpose. At 1024 dims they are
-- ~82MB/run, read exactly once by the prefilter and then dead. Inline, they TOAST-
-- bloat every SELECT * on candidates. Kept (rather than discarded) only so the
-- prefilter threshold can be re-tuned without re-embedding -- hence the TTL.
CREATE TABLE candidate_embeddings (
    candidate_id uuid PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE,
    run_id       uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    embedding    vector(1024) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON candidate_embeddings (run_id);
CREATE INDEX ON candidate_embeddings (created_at);

-- No HNSW/IVFFlat index on any vector column, deliberately. pgvector only uses an
-- approximate index when the query is ORDER BY <distance> LIMIT n. The prefilter is
-- a threshold gate over the whole run with no top-k, so the index would never be
-- consulted -- and approximate recall is disqualifying for a recall-critical gate
-- anyway. The prefilter runs as a numpy matmul in the worker (48ms for 20k x 1024
-- against ~300 leaves); these columns are storage, not a query surface.

-- one url can answer several leaves
CREATE TABLE candidate_nodes (
    candidate_id uuid NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    node_id      uuid NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    cosine       real NOT NULL,
    PRIMARY KEY (candidate_id, node_id)
);

CREATE TABLE scores (
    candidate_id uuid NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    node_id      uuid NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    is_seeking   bool NOT NULL,
    pain_match   int NOT NULL,
    icp_match    int NOT NULL,
    actionable   bool NOT NULL,
    reason       text,
    reply_angle  text,
    model        text NOT NULL,
    final        real NOT NULL,
    PRIMARY KEY (candidate_id, node_id)
);
CREATE INDEX ON scores (node_id, final DESC);

-- Calibration artifacts. The prefilter threshold is stored as a PERCENTILE, not a
-- raw cosine: absolute cosine values shift with model, input_type, dimension, and
-- dtype, so a raw cutoff silently means something different after any of those
-- change. A percentile survives the swap.
CREATE TABLE prefilter_config (
    id                  int PRIMARY KEY DEFAULT 1,
    model               text NOT NULL,
    input_type_query    text,
    input_type_document text,
    output_dimension    int NOT NULL,
    output_dtype        text NOT NULL,
    keep_percentile     real NOT NULL,
    calibrated_at       timestamptz,
    notes               text,
    CONSTRAINT single_row CHECK (id = 1)
);
