# FlexLoRA Implementation Log (Phase 2)

*Last updated: 2026-01-08*

This document records the **technical rationale**, integration decisions, and **risk-managed fixes** made while implementing the FlexLoRA baseline in this repository.

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

## 2) Safe Extend Strategy (shared utilities)

### 2.1 Non-negotiable safety requirement
The project contains multiple baselines that share core utilities (`general.py`, `helper.py`, ...). We therefore treat **regression risk** as a first-class constraint.

**Rule:** Fixes for FlexLoRA must be **isolated by explicit branching** (e.g., `method == "flex_lora"`) or implemented in FlexLoRA-specific modules.

### 2.2 Decision: extend `general.set_params` in an isolated branch
We extended `mak/utils/general.py::set_params` **only** for the FlexLoRA path:

- Trigger condition: `method == "flex_lora"`
- Mapping rule: payload must be mapped using `get_ffa_target_keys(model)` (single source of truth)
- Rank rule: LoRA factors `.A/.B` are sliced/padded to the client’s `local_rank` using `rank_map[client_id]`

All other methods/baselines keep the original behavior.

### 2.3 Why this is safe
- No changes to the default code path.
- No heuristic-based key ordering for FlexLoRA.
- This prevents silent corruption and minimizes blast radius.

---

## 3) Resource Mocking (rank distribution)

### 3.1 Context
We do not yet have the full upstream resource scheduler logic from the original author.

### 3.2 Decision
We simulate heterogeneous ranks deterministically via `config.yaml`:

- `flex_lora_config.rank_distribution`: list of `{rank, ratio}`
- `flex_lora_config.global_rank`: server/global rank
- `flex_lora_config.seed`: reproducible sampling seed

Example:
- 30% clients use rank 8
- 70% clients use rank 64 (global)

---

## 4) Protocol commitment (architecture mirroring)

- We keep the **Deterministic Name-Based Sorted List** protocol (mirrors FFA-LoRA integration).
- Payload keys are defined by `get_ffa_target_keys(model)`.
- Parameter injection stays centralized via `general.set_params`.

---

## 5) Fix Log (Aggregation Math Bug)

### 5.1 Problem
The naive approach “aggregate A and B separately” is mathematically incorrect:

- **Rejected:** `Avg(A)` and `Avg(B)`
- Reason: destroys correlation between factors; does not preserve \(\Delta W = A B\)

### 5.2 Fix (Implemented in `mak/strategies/flex_lora_strategy.py::aggregate_fit`)
For each LoRA layer (pair of `.A` and `.B`):

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

Standard trainable params (bias/head/classifier) use classic FedAvg weighted average.

### 5.3 Protocol handling
- Round 1: server sends FULL `state_dict` (initialization)
- Round > 1:
  - client uplink: partial ordered list (target keys)
  - server downlink: partial ordered list (same target keys)

---

## 6) Single Source of Truth for payload keys

### 6.1 Helper Utils
We use `get_ffa_target_keys(model)` in `mak/utils/helper.py` as the **single source of truth** for selecting communicated parameters.

Filtering rules:
- LoRA factors: `.A`, `.B`
- Bias terms: `.bias`
- Head weights: `.weight` containing keywords (`classifier`, `head`, `fc`, `score`, `linear`)

Return value: `sorted(set(keys))` to guarantee determinism.

---

## 7) Critical bug & safe fix (Rank drift / size mismatch during smoke test)

### 7.1 Symptom (Colab smoke test)
During `smoke_test_FlexLoRA.ipynb` on Google Colab (Python 3.11 + Miniconda), we observed client crashes at round > 1 evaluation:

- Error: `RuntimeError: size mismatch ... copying param shape [*, 8] into [*, 64]`
- Reproducible when rank_map contained heterogeneous ranks (e.g., `{0: 64, 1: 8}`)

### 7.2 Root cause analysis
There are two independent issues that must be handled safely:

**(A) Partial payload mapping must be deterministic**
- FlexLoRA client/server define partial payload order via `get_ffa_target_keys(model)`.
- Using heuristic key lists (e.g., filtering by substrings like `"lin"`) can cause key order mismatch and incorrect tensor assignment.

**(B) Re-ranking adapters by re-running `apply_svd_to_model` is not reliable**
- `apply_svd_to_model` adapts only `torch.nn.Linear` layers.
- After the model is wrapped by `SVDAdapter`, the original `Linear` layers no longer exist.
- In Ray actors we observed logs like:
  - `Found 0 linear layers to adapt with SVD.`
- Therefore, calling `apply_svd_to_model` again to switch rank (64 -> 8) becomes a no-op, leaving some clients with global-rank adapters.

This explains why payload rank 8 could be loaded into a model still expecting rank 64.

### 7.3 Safe fix adopted (FlexLoRA-specific, minimal blast radius)
We adopted a **client-safe adapter reconfiguration** in `mak/utils/flex_lora_utils.py::ensure_local_rank_adapters`.

#### Constraint (author intent)
- **SVD must only happen on the server.**
- Clients must not run `torch.linalg.svd` (or any decomposition of base weights).

#### What went wrong with the previous approach
Attempting to convert rank by slicing `.A/.B` tensors and calling `load_state_dict` failed because:
- Rank is an **architecture-level** property (Parameter shapes differ).
- You cannot load a `[*, 8]` tensor into an existing Parameter allocated as `[*, 64]`.
- Re-running `apply_svd_to_model` on the client is not allowed (client-side SVD) and is also unreliable once the model is already wrapped by `SVDAdapter`.

#### Final safe approach
When a client needs `local_rank` adapters:

- If the model already has adapters but with a different rank:
  - **Rebuild adapter modules** to match `local_rank` using **no-SVD LoRA init**:
    - `A`: small Gaussian
    - `B`: zeros
  - This creates the correct-shaped trainable Parameters for the client.
  - The server payload then overwrites these initial values.

- If the model has no adapters:
  - Inject adapters with the same **no-SVD** initialization.

This approach is isolated to FlexLoRA utilities and avoids touching shared SVD injection logic used by other baselines.

---

## 8) Operational notes for reproducibility

- When testing on Colab + Ray, always restart runtime after pulling code to avoid module caching.
- Always keep `rank_map` logged to confirm expected heterogeneous rank assignment.

*End of log.*
