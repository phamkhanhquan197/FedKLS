# FFA-LoRA Implementation Log

*Last updated: 2026-01-05*

## 0. Key timeline

| Date | Event |
|------|-------|
| 2025-12-25 | Initial FFA-LoRA scaffold completed (unstable). |
| 2025-12-31 | Refactor #1 – removed `requires_grad`-based mapping. |
| 2026-01-01 | Refactor #2 – switched to **Deterministic Name-Based Sorting**. |
| 2026-01-04 | Pulled + validated on lab (integration test); Phase 1 logic verified. |
| 2026-01-05 | Documentation finalized for Phase 2 (FlexLoRA) handoff. |

---

## 1. Current architecture (Phase 1)

| Component | File | Main responsibility |
|----------|------|---------------------|
| **Client** | `mak/clients/ffa_lora_client.py` | Filters uplink tensors (mainly `.B`, optional bias), **sorts keys deterministically**, uploads as `List[NDArrays]`. Loads FULL state_dict on Round 1 and partial updates on later rounds (client-side mapping). |
| **Strategy** | `mak/strategies/ffa_lora_strategy.py` | Inherits **FedAvg** and keeps server-side logic minimal: aggregates exactly what clients send (weighted avg), no key inspection/injection. |
| **Server wrapper** | `mak/servers/ffa_lora_server.py` | Wrapper for logging/history persistence; does not change baseline logic. |
| **Adapter math** | `mak/models/svd_model.py` | Adapter computes \(\Delta W = A \times B\). In FFA-LoRA, **A is frozen** (after init). |
| **Shared utils** | `mak/utils/general.py` | `set_params` performs safe parameter loading/mapping; evaluation (`test`) and metric aggregation (`weighted_average`). |
| **Init/SVD wiring** | `mak/utils/helper.py::apply_svd_to_model` | Builds adapter layers for the selected method; for `ffa_lora` initializes A/B and freezes A. |

---

## 2. Key configuration knobs (FFA-LoRA)

| Key | Default | Notes |
|-----|---------|------|
| `peft.method` | `ffa_lora` | Enables the FFA-LoRA pipeline. |
| `peft.rank` | 32 | Global rank for adapters. |
| `ffa_lora_config.init_method` | `kaiming` | `kaiming | gaussian | orthogonal | svd`. |
| `peft.bias` | `False` | If `True`, client also uploads selected bias terms (architecture-dependent keyword filter). |

---

## 3. Phase 1 Post-Mortem & Lessons Learned

### 3.1 Root cause of the previous failure

**`requires_grad`-based filtering/mapping was incorrect for the FL protocol.**

- The server tried to infer adapter/backbone tensors by filtering trainable parameters (`requires_grad=True`).
- After Round 1, **A is frozen**, which changes the trainable set.
- This caused server/client to disagree on payload length/order → **size/length mismatch**.

### 3.2 The successful fix

1. **Explicit Name-Based Sorting (deterministic mapping)**
   - Use explicit parameter naming rules (e.g., `.B`, optional bias/head weights).
   - Always **sort keys** before packing tensors into `List[NDArrays]`.

2. **Keep the server strategy simple**
   - Server aggregates what it receives (FedAvg-style) and does **not** attempt name-based injection into a full `state_dict`.

3. **Role of `mak/utils/general.py`**
   - Reliable parameter loading depends on reusing `general.set_params` to map incoming tensors back to the correct tensors.
   - Avoiding per-client hard-coded mapping was critical for stability.

### 3.3 Key lessons

| # | Lesson |
|---|--------|
| 1 | **Determinism first; optimization later.** Solve protocol correctness before bandwidth optimizations. |
| 2 | **Strict inheritance helps maintainability.** Do not modify shared bases (`BaseClient`, `ServerSaveData`). |
| 3 | **Math ↔ code alignment matters.** Factors used in SVD/init must match `svd_model.py`’s \(\Delta W = A \times B\). |

---

## 4. Lab verification (Integration Test)

> Current stage focuses on **protocol and system stability** (bug-fix verification). We intentionally avoid reporting numeric accuracy/loss values here.

### 4.1 Integration Verification Checklist (Round 1)

| Item | PASS criteria | Evidence from logs | Status |
|------|---------------|-------------------|--------|
| **Protocol Handshake** | Client receives the **FULL state_dict** on Round 1 | `Client 8: Loaded FULL state_dict (len=140).` | **PASSED** |
| **Safety Mechanism** | Client activates the **Safety Lock** for A (freeze A if trainable) | `Safety-lock applied (froze 0 A-params if they were trainable).` | **PASSED** |
| **Aggregation Logic** | Server aggregates tensors without any dimension/shape mismatch | `Aggregated Tensor 136: shape (768, 768)`; `Aggregated Tensor 137: shape (768,)` | **PASSED** |
| **Injection / Reload Logic** | Client successfully reloads parameters after aggregation (no crash due to length/shape mismatch) | Training continues into validation stage (no mid-round crash) | **PASSED** |
| **System Stability** | System completes Round 1 (train + evaluate) without crashing | Validation logs appear at end of round | **PASSED** |

### 4.2 Conclusion

**PHASE 1 LOGIC VERIFIED – READY FOR PHASE 2 (FlexLoRA).**

---

## 5. Next steps

- Start Phase 2 (FlexLoRA) implementation strictly following:
  - the **safe extension policy** for `mak/utils/general.py::set_params`
  - the **rank adaptation API contract** (slice/pad)
  - the **Deterministic Name-Based Sorted List** protocol

See: `docs/plan_sonvt8_ffa_flex_lora/Implementation_Plan_FFA_FlexLoRA.md`
