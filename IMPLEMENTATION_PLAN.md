# SAGE implementation plan and design

## 1. Objective

For a resident cache set \(C\) and a bounded recent-request window \(W\), SAGE maximizes:

\[
F(C)=\sum_{q\in W} w(q)\,\mathbf{1}[q\text{ is covered by at least one }e\in C].
\]

A resident covers a request when their prompt-embedding distance is no larger than the configured semantic-hit threshold. In the hit-rate policy, \(w(q)\) is only a recency weight. Response size, content category, UUIDs, and estimated answer cost are intentionally absent because they are not part of the optimized metric.

For a missed candidate \(x\) and resident victim \(e\), the exact replacement value is:

\[
\Delta(x,e)=F(C-\{e\}+\{x\})-F(C).
\]

The candidate is admitted only when the best replacement has a delta greater than the configured admission margin. Ties and zero-gain replacements are rejected to avoid churn.

## 2. Framework integration

### New single-pass lookup contract

`CacheLookup` is a Pydantic model with:

- `hit`
- `response`
- policy-specific `context`
- diagnostic `metadata`

`ICache.lookup()` has a backward-compatible default implementation that invokes the old `is_hit()` and `on_hit()` methods. SAGE overrides it to embed and search exactly once.

`EchoLLM.ask()` now calls `lookup()`. On a miss, its context is forwarded to `on_miss()`, allowing SAGE to reuse the already-computed embedding after the LLM returns.

### Compatibility

Old policies do not need to change. They continue through the fallback lookup implementation. SAGE also implements legacy `is_hit()` and `on_hit()` methods with thread-local lookup reuse for code that calls the old interface directly.

## 3. SAGE components

### `SAGESimilarityCache`

Coordinates lookup, admission, persistence, concurrency, statistics, and deterministic tie-breaking.

### `VectorSpace`

Provides exact vectorized cosine, Euclidean, and Manhattan distance. Cosine vectors are normalized on insertion/observation, turning lookup into a matrix-vector product.

### `GhostWindow`

A bounded circular buffer containing:

- recent request vectors;
- request arrival steps;
- a Boolean request-by-resident coverage matrix;
- coverage counts per request.

It contains no responses and cannot serve a hit. Rejected misses remain in the window, so repeated uncovered demand can build admission evidence.

### `SAGEScorer`

Calculates all replacement deltas without recomputing the full objective for every victim.

The candidate's common new gain is the weighted sum of requests that are currently uncovered and that the candidate covers. A victim's loss is the weighted sum of requests uniquely covered by that resident and not recovered by the candidate:

\[
\Delta(x,e)=\operatorname{NewGain}(x)-\operatorname{UnrecoveredLoss}(e,x).
\]

This is mathematically equal to the direct counterfactual objective. Randomized tests compare the vectorized formula against brute-force replacement.

### `SAGEStorage`

The policy performs lookups entirely in memory. Optional SQLite persistence is touched only on admission/replacement. Namespaces isolate experimental runs and deployments. Stored metadata prevents reopening a namespace with an incompatible vector dimension, distance method, or hit threshold.

All runtime configuration and transport/state records are Pydantic models; no dataclasses are used.

## 4. Request lifecycle

### Hit path

1. Embed the request once.
2. Compute exact distances to active residents.
3. Record the request and all covering residents in the ghost window.
4. Return the closest covering resident's response.
5. Update only the selected resident's last-access step for deterministic tie-breaking.

### Miss path

1. Record the miss in the ghost window during lookup.
2. Call the LLM outside SAGE's lock.
3. Reuse the embedding from `CacheLookup.context` when the LLM completes.
4. Recheck the current residents to prevent concurrent duplicate admission.
5. Admit directly when a free slot exists.
6. Otherwise evaluate the candidate against every possible victim.
7. Replace the best victim only for strictly positive net gain.

### Tie-breaking

Among victims with equal net gain:

1. least recently selected resident;
2. smallest stable slot index.

LRU is only a tie-breaker and does not alter the coverage objective.

## 5. Complexity

Let:

- \(C\): resident capacity;
- \(W\): ghost capacity;
- \(d\): embedding dimension.

Lookup:

\[
O(Cd)
\]

Full-cache miss admission:

\[
O(Wd+WC)
\]

The \(Wd\) term computes candidate-to-window distances; the \(WC\) representation is the coverage matrix, while score aggregation itself is vectorized and effectively \(O(W+C)\).

Memory:

\[
O(Cd+Wd+WC).
\]

The exact implementation is deliberate: it provides a trustworthy research baseline. If profiling later requires optimization, the coverage matrix can be bit-packed and resident lookup can use an ANN candidate stage while retaining exact reranking.

## 6. Concurrency and failure safety

- A re-entrant lock protects in-memory policy state.
- The lock is never held while the backend LLM runs.
- Miss completion rechecks current coverage, handling simultaneous semantically similar misses.
- Slot mutation is rolled back if persistence fails.
- Fixed resident slots keep ghost-matrix columns stable across replacement.
- `reset()` clears persistent state by default, avoiding an in-memory/storage split.

## 7. Verification plan

Automated tests cover:

1. vectorized versus brute-force deltas;
2. rejecting one-hit noise;
3. learning from repeated rejected misses;
4. exact hit behavior;
5. Euclidean support;
6. SQLite restart restoration;
7. one embedding per EchoLLM request;
8. OASST1 best-ranked response selection;
9. quality-adjusted metric denominator.
10. exact best-delta victim selection;
11. full chronological trace construction;
12. zero and automatically resolved unbounded capacities;
13. framework-native `ILLM.ask()` latency and latency percentiles;
14. identical memoized `LLMResponse` replay across policies.

Recommended additional large-run checks:

- phase-shift adaptation with and without decay;
- ghost-capacity sensitivity;
- admission-margin sensitivity;
- exact versus sampled ghost scoring;
- full chronological OASST1 runs from zero through unbounded capacity;
- response-quality evaluation with at least one independent evaluator model.
