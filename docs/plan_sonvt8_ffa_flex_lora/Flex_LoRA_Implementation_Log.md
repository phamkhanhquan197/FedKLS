# FlexLoRA Implementation Log (Baseline)

*Last updated: 2026-01-09*

This document records the **final baseline design**, key engineering decisions, and the **bugs/fixes** applied to make FlexLoRA stable in this repository.

> Contract: **SVD must run on server-side only.** Clients must never perform SVD/decomposition.

---

## 1) Adapter math (codebase ground truth)

The adapter implementation in `mak/models/svd_model.py` uses:

- \(W_{eff} = W_{res} + \frac{\alpha}{r}(A B)\)
- \(\Delta W = A B\)

Therefore server-side aggregation must preserve the semantics of **`A @ B`** (not averaging A/B independently).

---

## 2) Baseline protocol (Round 1 vs Round > 1)

### 2.1 Payload key contract (single source of truth)
We use `mak/utils/helper.py::get_ffa_target_keys(model)` as the **only** definition of communicated tensors.

Rules:
- LoRA factors: `.A`, `.B`
- Bias terms: `.bias`
- Head weights: `.weight` containing keywords: `classifier`, `head`, `fc`, `score`, `linear`

Keys are returned as `sorted(set(keys))` for determinism.

### 2.2 Round 1 (initialization)
- Server (strategy): sends **FULL** `state_dict` as a flat list in `state_dict().values()` order.
  - Implemented by `mak/strategies/flex_lora_strategy.py::initialize_parameters`.
- Client: receives FULL payload and must adapt global-rank parameters to its **local rank**.
  - Implemented by `mak/clients/flex_lora_client.py::set_parameters`:
    1) `ensure_local_rank_adapters(model, local_rank)` (architecture-level rank adaptation, no SVD)
    2) `slice_and_load_params(full_payload, local_rank)` (slice/pad A/B before loading)

### 2.3 Round > 1 (train loop)
- Client uplink: sends **PARTIAL** list aligned with `get_ffa_target_keys(model)`.
  - Implemented by `FlexLoRAClient.get_parameters`.
- Server downlink: returns **PARTIAL** list aligned with the same `get_ffa_target_keys(server_model)`.
- Client applies partial update via `mak/utils/general.py::set_params(method="flex_lora")`:
  - maps by `get_ffa_target_keys(model)`
  - slices/pads `.A/.B` to `rank_map[client_id]`

---

## 3) Server aggregation math (critical correctness)

### 3.1 Why “Avg(A) and Avg(B)” is wrong
Averaging factors independently destroys factor correlation and does not preserve \(\Delta W = AB\).

### 3.2 Correct aggregation (implemented)
For each LoRA layer pair `(A, B)`:

1) Client update: \(\Delta W_i = A_i B_i\)
2) Weighted average: \(\Delta W_{agg} = \sum_i w_i \Delta W_i\)
3) SVD on server: \(\Delta W_{agg} = U S V^T\)
4) Energy-preserving reprojection:
   - \(A_{new} = U\sqrt{S}\)
   - \(B_{new} = \sqrt{S}V^T\)

Standard (non-LoRA) communicated params (bias/head) are aggregated by weighted average.

Implementation: `mak/strategies/flex_lora_strategy.py::aggregate_fit`.

---

## 4) Rank heterogeneity (mocking)

We simulate heterogeneous client ranks deterministically via `config.yaml`:

- `flex_lora_config.global_rank`: server/global rank
- `flex_lora_config.rank_distribution`: list of `{rank, ratio}`
- `flex_lora_config.seed`

Rank map generation: `mak/utils/flex_lora_utils.py::generate_rank_map`.

---

## 5) Critical bug fixes (smoketest-driven)

### 5.1 Fix: CPU vs CUDA device mismatch in adapter residual (`W_res`)

**Symptom** (older run):
- `RuntimeError: Expected all tensors to be on the same device, but found cuda:0 and cpu!`
- at `SVDAdapter.forward`: `effective_weight = self.W_res + scaling * (A @ B)`

**Root cause**:
- `W_res` was not moving with `model.to(device)`.

**Fix** (server+client safe):
- Register `W_res` as buffer in `mak/models/svd_model.py`:
  - `self.register_buffer("W_res", W_res.clone().detach())`

### 5.2 Fix: Rank drift / `size mismatch` when local_rank < global_rank (Option A)

**Symptom** (failed smoketest):
- Client with local rank 8 crashed on Round 1 FULL payload load:
  - `size mismatch ... copying [*, 8] into [*, 64]`
- Log showed:
  - `Adapter rank mismatch ... -> rebuilding adapters (no SVD).`
  - `Injected 0 LoRA adapters without SVD (rank=8).`

**Root cause**:
- Server converts `nn.Linear -> SVDAdapter` at startup (`apply_svd_to_model`).
- Previous client rebuild logic attempted to scan for `nn.Linear` to re-inject adapters.
- After adaptation, there are **no `nn.Linear` modules** left; rebuild replaced 0 layers.

**Fix (Option A, minimal blast radius, no client SVD)**:
- Rebuild existing **`SVDAdapter` modules** directly to desired local rank.
- Implemented in `mak/utils/flex_lora_utils.py`:
  - `_rebuild_svd_adapters_no_svd(...)` (replaces SVDAdapter(rank=R) -> SVDAdapter(rank=r))
  - `ensure_local_rank_adapters(...)` now:
    1) tries `_rebuild_svd_adapters_no_svd` first
    2) falls back to `_inject_lora_adapters_no_svd` only if no SVDAdapters exist

**Smoketest status**:
- PASS (2 rounds completed; `FL finished in ...` marker present).

---

## 6) Prioritized issues & mitigations (baseline + paper alignment)

This section ranks the most important remaining issues ("issues" instead of "bugs") from highest to lowest priority.

### P0) Missing client-type support (paper fidelity blocker)

**Why important**:
- Paper FlexLoRA models resource heterogeneity via **4 client types** (Table 1) and multiple **type distributions** (Figure 3).
- Current baseline only supports **one rank per client** (uniform rank across all adapted layers), which cannot represent **Type 3** (different ranks for attention vs FFN layers).

**Current behavior (baseline)**:
- `flex_lora_config.rank_distribution` generates `rank_map[cid] -> local_rank`.
- `ensure_local_rank_adapters` and `set_params(method="flex_lora")` apply that single `local_rank` to all `.A/.B` factors.

**Impact**:
- The implementation cannot reproduce key experiments/claims of the paper regarding heterogeneous resource distributions.
- Any evaluation of FlexLoRA under heterogeneous client types is currently incomplete.

**Proposed direction (short)**:
- Replace `rank_map` with a richer **client configuration map**:
  - `client_type_map[cid] -> type_id`
  - `type_id -> rank_policy`, e.g.
    - Type1: `r=8` all tunable layers
    - Type2: `r=30` all tunable layers
    - Type3: `r_attn=30`, `r_ffn=200` (MAM-style)
    - Type4: `r=200` all tunable layers
- Add layer-group detection for models (at least DistilBERT/BERT-style):
  - attention layers: names containing `attention` or `self_attn`
  - FFN layers: names containing `ffn`, `mlp`, `lin1/lin2`
- Extend slicing/loading to be **per-layer rank** (not single rank).

**Minimal smoke tests to add**:
- 2 clients, 2 rounds, assign Type3 to at least one client, verify:
  - no `size mismatch` on Round 1 full payload
  - communication size differs by type (Type3 larger than Type1)
  - aggregation runs without crash

### P1) Repeated local-rank rebuild can silently reset adapter state

**Why important**:
- Smoke logs show frequent `Adapter rank mismatch ... -> rebuilding adapters (no SVD)` events for low-rank clients.
- Current rebuild path re-initializes `A` (random) and `B` (zeros), which can erase learned adapter state and degrade convergence without a crash.

**Symptom (smoke logs)**:
- `Adapter rank mismatch detected. desired=8 ... -> rebuilding adapters (no SVD).`
- `Rebuilt 36 SVDAdapter modules without SVD (rank=8).`

**Proposed mitigation (short)**:
- Make rank adaptation **idempotent and preserving**:
  - When changing rank, project existing `(A,B)` by truncate/pad rather than re-init.
  - Ensure `ensure_local_rank_adapters` runs only once per client lifecycle where possible.

### P2) Payload protocol relies on list length (fragile with heterogeneous client types)

**Why important**:
- The code infers "full vs partial" by `len(payload)`.
- With future client-type extensions (different communicated keys), this can break aggregation or cause hard-to-debug mismatches.

**Proposed mitigation (short)**:
- Carry explicit metadata in config (e.g., `payload_kind: full|partial`) or include a stable header.
- Validate keys against a shared contract per client type.

### P3) Server-side SVD scalability and OOM risk (large models)

**Why important**:
- `torch.linalg.svd` per-layer on `ΔW_agg` can be expensive for large hidden sizes or many layers.

**Proposed mitigation (short)**:
- Consider truncated/randomized SVD or CPU fallback.
- Add guardrails/logging for per-layer SVD time and memory.

### P4) Conv/vision models may need special handling

**Why important**:
- Current FlexLoRA aggregation assumes 2D factors merged by `A@B`.
- For Conv adapters, ensure factorization/reshape contracts are consistent end-to-end.

**Proposed mitigation (short)**:
- Add explicit tests for a CNN backbone (e.g., ResNet) if FlexLoRA is expected to support it.

### P5) Warning: Flower reports "Both server and strategy were provided, ignoring strategy"

**Why important**:
- This is a compatibility/maintenance risk across Flower versions.
- Even if the current run works, behavior could change.

**Proposed mitigation (short)**:
- Ensure the simulation is configured in one canonical way (either pass `server=` or rely on strategy-managed server).

---

## 7) Smoke test harness notes

In `smoke_test_FlexLoRA.ipynb`, the failure detector should grep for generic mismatch messages.

Recommended check:
- grep for `"size mismatch"` (Torch often prints `size mismatch for <param> ...`)
- not only `"RuntimeError: size mismatch"`.

---

## 8) File map (baseline implementation)

- Entry: `main.py`
  - generates `rank_map` when `strategy == FlexLoRA`
  - applies server-side SVD adaptation using `apply_svd_to_model` with `global_rank`
  - deep-copies model per client to avoid shared mutation
- Utilities: `mak/utils/flex_lora_utils.py`
  - rank_map generation, server-rank config, local-rank rebuild/injection, full-payload slicing
- Client: `mak/clients/flex_lora_client.py`
  - Round 1 full load with local-rank architecture ensure
  - Round > 1 partial updates using deterministic keys
- Server Strategy: `mak/strategies/flex_lora_strategy.py`
  - correct LoRA aggregation by ΔW-space + SVD reprojection (server-side only)
- Server wrapper: `mak/servers/flex_lora_server.py` (thin wrapper)

*End of log.*
