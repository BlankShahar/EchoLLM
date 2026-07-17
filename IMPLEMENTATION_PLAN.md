# W-SAGE implementation design

## Objective

W-SAGE keeps EchoLLM's binary semantic-hit rule unchanged:

\[
d(q,e) \le \tau.
\]

Only admission and eviction scoring changes. For main-cache residents \(M\),
demand observations \(Q\), recency weights \(w_q\), and configurable utility
kernel \(K\):

\[
F(M)=\sum_{q\in Q}w_q\max_{e\in M}K(d(q,e)).
\]

Binary mode uses \(K(d)=1[d\le\tau]\). Soft mode, enabled by default, uses:

\[
K(d)=1[d\le\tau]\left(\max(0,1-d/\tau)\right)^\gamma.
\]

Thus closer semantic representatives are preferred, but a request cannot become
a hit merely because soft scoring is enabled.

## Cache partition

The fixed slot array is split into:

- an LRU probation window, default 5% of total capacity;
- a SAGE-managed main cache.

For capacities greater than one, a non-zero window always receives at least one
slot and always leaves at least one main slot. Capacity one becomes a pure LRU
probation cache. Setting `window_fraction=0` provides the direct-admission SAGE
ablation.

## Request lifecycle

### Lookup

1. Embed the prompt exactly once.
2. Scan active window and main residents using the configured exact distance.
3. Return the closest resident satisfying the shared binary hit threshold.
4. Record demand utilities only against main-cache residents.
5. Update the selected resident's access time.

### Miss and probation

1. EchoLLM invokes the backend outside the cache lock.
2. W-SAGE reuses the embedding carried in `CacheLookup.context`.
3. It rechecks all residents to prevent concurrent semantic duplicates.
4. The incoming prompt/response is inserted into the probation window.
5. If the window is full, its LRU resident becomes the main-cache promotion
   candidate.

The current miss therefore receives temporary residency without forcing itself
into the long-lived main cache.

### Promotion

If main has a free slot, the exiting window resident is promoted. Otherwise, for
every main victim \(e\), W-SAGE evaluates:

\[
\Delta(x,e)=F(M-\{e\}+\{x\})-F(M).
\]

The candidate is promoted only when the best delta exceeds the admission margin.
The incoming item still remains in probation when the exiting candidate is
rejected.

## Exact soft counterfactual scoring

Each evidence row maintains its highest and second-highest resident utility. For
a candidate utility \(u_x\), the common gain is:

\[
\sum_q w_q(\max(u_x,u_1)-u_1).
\]

Only the unique top resident of a row can incur additional replacement loss;
its corrected post-replacement utility is \(\max(u_x,u_2)\). These corrections
are aggregated by owner with vectorized NumPy operations. Randomized tests compare
the optimized result against brute-force replacement for every victim.

## Recent and long-term evidence

The recent ring stores every observation. When observations age out, every
`long_sample_stride`-th eviction is inserted into a separate long-term ring.
The two rings are therefore disjoint.

Each horizon has its own capacity, half-life, and normalized score. Their deltas
are combined with `recent_evidence_weight`. Capacities derive from main-cache
size using configurable multipliers and hard limits, preventing unbounded
request-by-resident state.

## Tie-breaking

Victims first maximize exact replacement delta. Equal-delta victims are compared
using fractional responsibility:

\[
R(e)=\sum_q w_q\frac{K(d(q,e))}{\sum_jK(d(q,j))}.
\]

The lowest-responsibility victim is evicted. LRU and stable slot order are used
only if both delta and responsibility tie. Per-column non-zero counts provide an
exact fast path for zero-responsibility residents; remaining responsibility
calculations are chunked to bound temporary memory.

## Consistency and persistence

Window rotation may move one key from a window slot to a main slot while placing
the incoming key into the vacated window slot. `SAGEStorage.apply()` writes all
changed slots in one SQLite transaction. In-memory vectors, keys, responses,
timestamps, and evidence columns are rolled back together if persistence fails.

Stored metadata includes vector/distance settings, window size, and soft-coverage
settings. Reopening an incompatible namespace fails explicitly.

## Experiment datasets

OASST1 follows one deterministic root-to-leaf path per conversation tree. At a
prompter, the selected assistant child is ordered by rank, quality, timestamp,
and stable ID; at an assistant, the earliest eligible prompter continuation with
a usable response is selected. Every emitted prompt has one response.

WildChat scans conversations as a stream, extracts adjacent user-to-assistant
turns, and keeps the globally earliest configured number of assistant-timestamped
requests using a bounded max-heap. The shipped configuration retains 50,000
requests and uses the current user message without conversation context.

## Validation invariants

- Window and main masks are disjoint and cover total capacity.
- Cache size never exceeds total capacity.
- Window columns never contribute to the SAGE objective.
- Recent and long observations are disjoint.
- Utilities remain in `[0, 1]`.
- Lookup uses one prompt embedding and the closest qualifying resident.
- Incoming probation insertion and optional main promotion are atomic.
- Optimized soft deltas equal brute-force deltas.
- `soft_coverage=False` retains binary coverage behavior.
- Every policy/capacity experiment starts with a new in-memory cache.
