# FlexLoRA Implementation Log (Phase 2)

*Last updated: 2026-01-06*

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

*End of log.*

