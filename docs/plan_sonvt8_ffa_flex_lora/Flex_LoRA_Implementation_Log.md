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

## 6) Smoke test harness notes

In `smoke_test_FlexLoRA.ipynb`, the failure detector should grep for generic mismatch messages.

Recommended check:
- grep for `"size mismatch"` (Torch often prints `size mismatch for <param> ...`)
- not only `"RuntimeError: size mismatch"`.

---

## 7) File map (baseline implementation)

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
