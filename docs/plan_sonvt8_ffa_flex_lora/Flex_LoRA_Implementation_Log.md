# FlexLoRA Implementation Log (Phase 2)

*Last updated: 2026-01-08*

This document records the **technical rationale** and non-negotiable decisions made before implementing the FlexLoRA baseline in this repository.

---

## 1) Math consensus (SVD + adapter fidelity)

### 1.1 Adapter math in this codebase
The adapter implementation in `mak/models/svd_model.py` uses:

- **Effective weight**:  \(W_{eff} = W_{res} + \text{scaling} \cdot (A \times B)\)
- Therefore the low-rank update is: \(\Delta W = A \times B\)

This means our server-side reconstruction/aggregation MUST preserve the semantics of **`A @ B`**.

### 1.2 Server-side SVD merge decision
For each round, the server will:

1) Reconstruct each client update as \(\Delta W_i = A_i \times B_i\)
2) Weighted-average to obtain \(\Delta W_{agg}\)
3) Perform SVD on the aggregated update:

\[
\Delta W_{agg} = U S V^T
\]

### 1.3 Energy-preserving re-projection (critical)
We explicitly avoid sending raw singular vectors without scale.

- **Rejected**: \(A = U\), \(B = V^T\)  
  This loses magnitude information contained in \(S\).

- **Accepted** (energy split across factors):
\[
A = U \sqrt{S}, \qquad B = \sqrt{S} V^T
\]

So that:
\[
A B = U \sqrt{S} \sqrt{S} V^T = U S V^T
\]

This matches the adapter update form \(\Delta W = A \times B\) used in `svd_model.py`.

---

## 2) Safe Extend Strategy (general.py)

### 2.1 Non-negotiable safety requirement
We **do not rewrite** `mak/utils/general.py::set_params`. We only **extend** it in a backward-compatible way.

### 2.2 Decision
- Extend `set_params` signature to accept:
  - `method: Optional[str] = None`
  - `rank_map: Optional[Dict[int, int]] = None`
  - `client_id: Optional[int] = None`
- **Only** when `method == "flex_lora"` will rank slice/pad logic be activated.
- For all existing methods/baselines (`FedAvg`, `Scaffold`, `FedNova`, `FFALoRA`, etc.), behavior remains unchanged.

### 2.3 Why this is safe
- The branching condition is explicit (`method == "flex_lora"`).
- The default path is identical to existing logic.
- This prevents regressions in other baselines using the shared `set_params`.

---

## 3) Resource Mocking (rank distribution)

### 3.1 Context
We do not yet have the full upstream resource scheduler logic from the author.

### 3.2 Decision
We will simulate heterogeneous resources using a deterministic random assignment from `config.yaml`:

- `flex_lora_config.rank_distribution`: list of `{rank, ratio}`
- `flex_lora_config.global_rank`: the maximum server rank
- Assignment uses `flex_lora_config.seed` to ensure reproducibility.

Example:
- 30% clients use rank 8
- 70% clients use rank 64 (global)

---

## 4) Protocol commitment (architecture mirroring)

- We keep the **Deterministic Name-Based Sorted List** protocol from Phase 1.
- Client payload selection is done **inline** (blueprint: `ffa_lora_client.py`).
- Parameter injection stays centralized via `general.set_params`.

---

## 5) Fix Log (Aggregation Math Bug)

### 5.1 Problem
The previous FlexLoRA server aggregation was mathematically incorrect when it **aggregated A and B separately** (e.g., `Avg(A)` and `Avg(B)` via FedAvg-style aggregation).

Reason: averaging factors independently destroys the correlation between A and B and does **not** preserve the correct low-rank update semantics used by this repo:

\[
\Delta W = A \times B
\]

### 5.2 Fix (Implemented in `mak/strategies/flex_lora_strategy.py::aggregate_fit`)
We changed the aggregation rule to:

- **Rejected:** `Avg(A)` and `Avg(B)`
- **Accepted:** `SVD(Avg(A @ B))`

Concretely, for each LoRA layer (pair of `.A` and `.B`):

1) Reconstruct each client update:
\[
\Delta W_i = A_i B_i
\]

2) Weighted-average in matrix space (layer-wise, memory-safe):
\[
\Delta W_{agg} = \sum_i \frac{n_i}{\sum_j n_j} \Delta W_i
\]

3) SVD merge and energy-preserving reprojection:
\[
\Delta W_{agg} = U S V^T,\quad
A_{new}=U\sqrt{S},\quad
B_{new}=\sqrt{S}V^T
\]

For **standard trainable params** (e.g., `.bias`, `classifier.weight`, ...), we keep **FedAvg weighted average**.

### 5.3 Protocol handling
- Round 1: server sends FULL `state_dict` (init)
- Round > 1:
  - client uplink: **all trainable params** in a partial ordered list
  - server downlink: **partial** ordered list (LoRA A/B after SVD-merge + standard params after weighted avg)

---

## 6) Final Integration (Helper + Protocol Freeze)

### 6.1 Helper Utils Update
- Added/confirmed `get_ffa_target_keys(model)` in `mak/utils/helper.py`.
- This function is the **Single Source of Truth** for selecting communication parameters across Client and Server.
- Filtering rules (as implemented):
  - LoRA factors: `.A`, `.B`
  - Bias terms: `.bias`
  - Head weights: `.weight` containing keywords (`classifier`, `head`, `fc`, `score`, `linear`)
- Return value is always `sorted(set(keys))` for determinism.

### 6.2 Protocol Finalization
- The final communication protocol is:
  - **Deterministic Sorted List** based on **keys returned by `get_ffa_target_keys(model)`**.
- This mirrors Phase 1 (FFA-LoRA) and prevents payload mismatch.

### 6.3 Code Freeze
- Client, Strategy, Server, Utils, Config have been aligned to the same protocol and math.
- This codebase is considered **ready for release** as the FlexLoRA Phase 2 “Golden Release”.

---

## 7) Smoke-test watchlist (known risks to validate)

This section tracks **known risks** that are relevant to the FlexLoRA baseline integration.
They are not necessarily bugs, but must be validated during the Google Colab smoke test.

### 7.1 Rank map wiring (High)
**Risk:** For FlexLoRA, each client needs a deterministic `rank_map[cid]` so that `set_params(..., method="flex_lora")` can slice/pad adapter factors correctly.

**What to validate in smoke test:**
- Ensure `rank_map` is passed end-to-end: `main.py` → `get_client_fn` → `FlexLoRAClient(rank_map=...)`.
- Confirm `FlexLoRAClient.set_parameters()` receives a non-empty `rank_map` and that `rank_map[client_id]` exists.

**Expected behavior:**
- Fail-fast is acceptable if rank wiring is missing (misconfiguration).

### 7.2 Payload key order consistency vs. `set_params` heuristics (Very High)
**Risk:** FlexLoRA strategy/client use `get_ffa_target_keys(model)` to define the partial payload order.
However, `mak/utils/general.py::set_params` currently derives the receiving key list via model-specific heuristics.

If these two key lists diverge, parameters may be assigned to wrong tensors (silent corruption) or raise shape/load errors.

**What to validate in smoke test:**
- Compare (once) `get_ffa_target_keys(model)` with the internal key list used by `set_params` for the same model.
- Validate that the number of tensors and their shapes align for partial updates (Round > 1).

**Expected behavior:**
- Under Strict Homogeneity, a mismatch indicates incorrect integration/config and should fail fast.

### 7.3 Resource / OOM risks in server-side SVD merge (Medium)
**Risk:** FlexLoRA server performs per-layer SVD of aggregated \(\Delta W\) each round.
This can be heavy for transformer backbones and may trigger OOM or slowdowns.

**What to validate in smoke test:**
- Monitor peak memory usage on the server process during aggregation.
- If OOM occurs, mitigate via test config (smaller model, fewer clients, smaller global rank, fewer rounds).

*End of log.*
