# FlexLoRA Implementation Log (Baseline)

*Last updated: 2026-01-10*

This document records the **final baseline design**, key engineering decisions, and the **issues/fixes** applied to make FlexLoRA stable in this repository.

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
- Client: receives FULL payload and must adapt global-rank parameters to its **local rank policy**.
  - Implemented by `mak/clients/flex_lora_client.py::set_parameters`:
    1) `ensure_local_rank_adapters(model, rank_policy)` (architecture-level rank-policy adaptation, no client SVD)
    2) `slice_and_load_params(full_payload, rank_policy)` (slice/pad A/B per adapter base)

### 2.3 Round > 1 (train loop)
- Client uplink: sends **PARTIAL** list aligned with `get_ffa_target_keys(model)`.
  - Implemented by `FlexLoRAClient.get_parameters`.
- Server downlink: returns **PARTIAL** list aligned with the same `get_ffa_target_keys(server_model)`.
- Client applies partial update via `mak/utils/general.py::set_params(method="flex_lora")`:
  - maps by `get_ffa_target_keys(model)`
  - slices/pads `.A/.B` per-adapter using `rank_policy` (Type 3 supported)

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

## 4) Resource heterogeneity (paper-aligned client types)

The paper models client resource heterogeneity via **4 LoRA configuration types** (Table 1) and multiple distributions (Figure 3).

In this baseline:
- `flex_lora_config.global_rank` is treated as **max/global rank** (paper Type 4, usually 200)
- Each client is assigned a **client type** and derives a **rank policy**

Type definitions (paper Table 1):
- Type 1: `r = 8` on all layers
- Type 2: `r = 30` on all layers
- Type 3: `r = 30` on attention layers, `r = 200` on FFN layers (MAM-style)
- Type 4: `r = 200` on all layers

Config knobs:
- `flex_lora_config.client_type_map`: explicit mapping `cid -> type_id` (recommended for deterministic smoke tests)
- `flex_lora_config.client_type_distribution`: list of `{type, ratio}` (for large-N simulations)

Implementation:
- `mak/utils/flex_lora_utils.py::build_client_type_map`
- `mak/utils/flex_lora_utils.py::build_client_rank_policy_map`

---

## 5) Prioritized issues & mitigations (baseline + paper alignment)

This section ranks the most important remaining issues ("issues" instead of "bugs") from highest to lowest priority.

### P0) Client-type support (paper fidelity)

**Status**: **PASS (smoke test)**

**What was missing**:
- Older baseline only supported a single `rank` per client, uniformly applied to all adapters.
- This could not represent paper **Type 3** (different ranks for attention vs FFN layers).

**Key changes that enabled PASS**:
- Introduced **client types** and derived **per-layer rank policies**:
  - `client_type_map[cid] -> type_id`
  - `rank_policy[cid] -> {all|attn|ffn: rank}`
- Enforced **global_rank = max rank** (paper Type 4): `global_rank = 200`, with local ranks `<= global_rank`.
- Refactored client-side adaptation and parameter loading to be **policy-based** (per adapter base):
  - `ensure_local_rank_adapters(model, rank_policy)`
  - `slice_and_load_params(full_payload, rank_policy)`
  - `set_params(method="flex_lora")` slices/pads `.A/.B` per adapter using `rank_policy`

**Smoke evidence (example run)**:
- `FlexLoRA global_rank: 200`
- `FlexLoRA client_type_map: {0: 3, 1: 1}`
- `FlexLoRA client_rank_policy_map: {0: {'attn': 30, 'ffn': 200}, 1: {'all': 8}}`
- `FL finished in ...` marker present.

**Notes**:
- The smoke test is a functional correctness gate (pipeline runs end-to-end with Type 3 present).
- Paper-scale experiments still require distribution-driven sampling (Uniform/Heavy-tail/Normal) on large client counts.

### P1) Repeated adapter rebuild (rank mismatch) can silently reset adapter state

**Status**: **IMPROVED (rebuild reduced)**

**What was the issue**:
- Smoke logs showed frequent `Adapter rank-policy mismatch ... -> rebuilding adapters (no SVD)` events, even in rounds after initialization.
- The original rebuild logic re-initialized adapters (`A=randn`, `B=zeros`), which could erase learned state and degrade convergence.

**Key changes that improved this**:
- **Step 1 (Preserve state)**: Rebuild logic in `_rebuild_svd_adapters_no_svd_policy` was changed to **project** existing `A` and `B` matrices (truncate/pad) instead of re-initializing. This ensures that even if a rebuild is triggered, learned weights are not lost.
- **Step 2 (Reduce frequency)**: A `_policy_initialized` flag was added to `FlexLoRAClient`. This ensures the expensive/disruptive `ensure_local_rank_adapters` is called only once during the first full payload update. Subsequent rounds skip this check, assuming the policy-shaped model is maintained.

**Smoke evidence (latest run)**:
- `Adapter rank-policy mismatch detected ...` log **only appears in Round 1** (fit and eval phases).
- **Round 2 shows no rebuild logs**, confirming the `_policy_initialized` flag is working.

**Remaining sub-issue**:
- Rebuild still occurs between `fit` and `eval` in Round 1. This is a minor issue now that state is preserved, but indicates a small state drift within the client actor's lifecycle that could be further optimized.

### P2) Payload protocol relies on list length (fragile with heterogeneous client types)

**Status**: **PASS (smoke test)**

**What was the issue**:
- The client logic used `len(parameters)` to differentiate between a full model update (Round 1) and a partial update (Round > 1).
- This is fragile and would break if different client types had different sets of trainable parameters, leading to different payload lengths.

**Key changes that enabled PASS**:
- **Explicit protocol via metadata**: The server now adds a `payload_kind: "full" | "partial"` field to the config dictionary sent to clients during `fit`.
- **Client-side logic update**: `FlexLoRAClient` was refactored to read this `payload_kind` from the config to determine how to handle the incoming parameters, removing the dependency on `len()`.
- **Backward compatibility**: A fallback to the `len()`-based logic was kept to ensure `BaseClient` behavior is not broken for other strategies.

**Smoke evidence (latest run)**:
- The smoke test passed successfully with the new protocol, confirming no regressions.
- The `TypeError` crash (caused by a temporary signature mismatch during refactoring) was resolved.

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

---

## 6) Critical bug fixes (smoketest-driven)

### 6.1 Fix: CPU vs CUDA device mismatch in adapter residual (`W_res`)

**Symptom** (older run):
- `RuntimeError: Expected all tensors to be on the same device, but found cuda:0 and cpu!`
- at `SVDAdapter.forward`: `effective_weight = self.W_res + scaling * (A @ B)`

**Root cause**:
- `W_res` was not moving with `model.to(device)`.

**Fix** (server+client safe):
- Register `W_res` as buffer in `mak/models/svd_model.py`:
  - `self.register_buffer("W_res", W_res.clone().detach())`

### 6.2 Fix: Round-1 `size mismatch` when local rank differs from global rank (historical)

**Symptom** (older run):
- Low-rank clients crashed on Round 1 FULL payload load:
  - `size mismatch ... copying [*, r_small] into [*, r_global]`

**Root cause**:
- Server converts `nn.Linear -> SVDAdapter` at startup (`apply_svd_to_model`).
- Older client logic attempted to scan for `nn.Linear` to re-inject adapters.
- After server adaptation there are no `nn.Linear` modules left.

**Fix**:
- Rebuild existing `SVDAdapter` modules without SVD on client.

**Current status**:
- PASS for the updated policy-based pipeline (no `size mismatch` in latest smoke test).

---

## 7) Smoke test harness notes

In `smoke_test_FlexLoRA.ipynb`, the failure detector should grep for generic mismatch messages.

Recommended check:
- grep for `"size mismatch"` (Torch often prints `size mismatch for <param> ...`)
- not only `"RuntimeError: size mismatch"`.

NOTE (important for bash cells):
- When using `set -e`, optional path-based checks should not return a non-zero exit code.
- Prefer a resilient log finder (e.g., `find /content/output -name log.txt`) and use `|| true` for optional checks.

---

## 8) File map (baseline implementation)

- Entry: `main.py`
  - builds `client_type_map` and `client_rank_policy_map` when `strategy == FlexLoRA`
  - applies server-side SVD adaptation using `apply_svd_to_model` with `global_rank`
  - deep-copies model per client to avoid shared mutation
- Utilities: `mak/utils/flex_lora_utils.py`
  - client-type assignment, per-layer rank policy, policy-based adapter rebuild and full-payload slicing
- Client: `mak/clients/flex_lora_client.py`
  - Round 1 full load with rank-policy architecture ensure
  - Round > 1 partial updates using deterministic keys + policy-based slice/pad
- Server Strategy: `mak/strategies/flex_lora_strategy.py`
  - correct LoRA aggregation by ΔW-space + SVD reprojection (server-side only)
- Server wrapper: `mak/servers/flex_lora_server.py` (thin wrapper)

*End of log.*
