# FFA-LoRA Implementation Log (Phase 1)

Date: 2025-12-25

## Scope implemented
- Phase 1: FFA-LoRA only (NO FlexLoRA logic).
- Target layers: **Conv2d** (ResNet conv2/conv1 selection as existing) + **Linear** layers (existing selection).
- Protocol: Flower **Round starts at 1**.
- Communication: **List[NDArrays]** with deterministic order.
- Constraint: **Do NOT modify** `SVDAdapter` / `ConvAdapter` classes; freeze A externally.

---

## Changes made

### 1) `config.yaml`
- Added new section `ffa_lora_config`:
  - `seed`
  - `init_method` (supports: `kaiming`, `gaussian`, `orthogonal`, `svd`)
- Updated comment in `peft.method` list to include `ffa_lora`.

### 2) `mak/utils/helper.py`
- Updated `apply_svd_to_model` to handle `method == 'ffa_lora'`.
- **CRITICAL FIX (SVD init correctness)**: initialize A using **left singular vectors** `U[:, :rank]` (shape `[d_out, rank]`) because `SVDAdapter` computes \(\Delta W = A @ B\).

Implemented initialization logic:
- **Matrix A** initialized by `ffa_lora_config.init_method`:
  - `kaiming`: `torch.nn.init.kaiming_normal_`
  - `gaussian`: `torch.nn.init.normal_(std=0.01)`
  - `orthogonal`: `torch.nn.init.orthogonal_`
  - `svd`: uses `U[:, :rank]` from SVD of weight matrix (Conv2d weights flattened to `[c_out, c_in*k*k]`).
- **Matrix B** initialized as **zeros**.
- **W_res** kept as original weight matrix (no residual subtraction for FFA-LoRA).

Freeze rule:
- After creating adapter layer, applied external freeze:
  - `new_layer.A.requires_grad = False`

Wiring:
- Added imports for `FFALoRAStrategy` and `FFALoRAServer`.
- Updated `get_server` to return `FFALoRAServer` when strategy is `FFALoRAStrategy`.
- Updated `get_strategy` kwargs mapping to provide `config` into the strategy constructor.

### 3) `mak/clients/ffa_lora_client.py`
- Client implementation for FFA-LoRA.

---

## Phase 1 Refactor (2025-12-31): requires_grad Index-Mapping + Size-Mismatch Fix

### Summary
- Refactored **FFA-LoRA Client/Strategy** to use **requires_grad-based index mapping**.
- Fixed critical `RuntimeError: size mismatch` by:
  - Server maintaining a **FULL global parameter snapshot** (`current_full_parameters`).
  - Server sending **FULL parameters** in round 1, then **trainable-only parameters** from round 2+.
  - Server aggregating **trainable-only tensors** via `super().aggregate_fit(...)` and then **reconstructing** a FULL parameter list before returning.
- Added **Safety Locks** to guarantee **Frozen-A** on clients (even if upstream config/protocol is wrong).

---

## Finalize Critical Fix (2026-01-01): Deterministic Name-Based Aggregation (Sorted Keys)

### Why this change
The previous `requires_grad`-based mapping approach still failed in lab testing due to potential nondeterminism/misalignment in parameter ordering between client/server and differences in how adapters expose trainable tensors.

### What we changed (final)
- **Switched to Deterministic Name-Based Mapping (Sorted Keys).**
  - Standardized all communication on **`model.state_dict()`**.
  - Introduced `get_ffa_target_keys(model) -> List[str]` shared by Client and Server.
  - Mapping pipeline: **Name Filter → Sort → Key-Based Injection**.
- **Aggregation scope (paper + author requirement):**
  - Aggregate only:
    - LoRA matrices: `*.B`
    - Bias terms for Transformer/CNN/heads: `*.bias` with keyword filter
    - Classifier/head weights: `*.weight` with head keywords (`classifier`, `head`, `fc`, `pre_classifier`)
- **Uplink Consistency:** Client `get_parameters` now **always returns PARTIAL** (based on `target_keys`) for every round, including Round 1.
- **Downlink Logic:**
  - Round 1: server sends FULL state_dict values
  - Round >1: server sends PARTIAL state_dict values (by `target_keys`)
- **Key-based injection implemented** on both client and server using in-place `.copy_()`.
- **SVD init corrected** for `ffa_lora` + `init_method="svd"`: A uses `U[:, :rank]` (left singular vectors) to match `SVDAdapter` math `A @ B`.

### Status
**PHASE 1 COMPLETE - READY FOR LAB TEST.**

---

## Files changed/added
- Modified:
  - `mak/utils/helper.py`
  - `mak/clients/ffa_lora_client.py`
  - `mak/strategies/ffa_lora_strategy.py`
  - `docs/plan_sonvt8_ffa_flex_lora/FFA_LoRA_Implementation_Log.md`
