# Implementation Plan (International Version)

*Last updated: 2026-01-05*

## 0. Non-negotiable rules (Lessons learned from Phase 1)

### 0.1 Strict inheritance (commitment)
- **Do not modify** `BaseClient` or `ServerSaveData`.
- **Do not copy/paste** core logic from `BaseClient` or Flower strategies; use `super()` as much as possible.
- All FlexLoRA logic must live in subclasses:
  - `FlexLoRAClient`
  - `FlexLoRAServer`
  - `FlexLoRAStrategy` (or equivalent)

### 0.2 Strict code reuse (commitment)
- **Do not create redundant helper APIs** for pack/unpack if the logic can be integrated into `mak/utils/general.py`.
- Reuse existing utilities in `mak/utils/helper.py`.
- The single "source of truth" for parameter injection/loading must be `mak/utils/general.py::set_params`.

### 0.3 Architecture mirroring (blueprint)
Treat Phase 1 files as the blueprint:
- `mak/clients/ffa_lora_client.py` for **inline filtering + deterministic sorting**
- `mak/strategies/ffa_lora_strategy.py` for **minimal server-side strategy**
- `mak/utils/general.py::set_params` for **robust parameter injection**

---

## 1. Phase 1 recap: FFA-LoRA (verified)

### 1.1 What worked
- Deterministic **Name-Based Sorted List** protocol
- Server-side strategy kept simple; client decides payload
- Safe parameter (re)loading using `general.set_params`

### 1.2 What to keep for Phase 2
- The communication protocol: **Deterministic Name-Based Sorted List**
- The "no base-class modification" rule
- Reuse of `general.set_params` for parameter injection

---

## 2. Phase 2: FlexLoRA — Engineering Specs

### 2.1 Key Differences from FFA-LoRA
- **Both A and B are trained and communicated** (FFA-LoRA only trained B)
- **Heterogeneous ranks** across clients (each client can have different rank budget)
- **SVD merge performed every round** (not just initialization)

### 2.2 Data Structure Specification

#### 2.2.1 Rank Specification (`rank_spec`)
```json
{
  "global_rank": 64,
  "local_rank": 8,
  "param_type": "A"  // or "B" to determine slice/pad dimension
}
```

#### 2.2.2 Tensor Naming Convention
- `.A`: Left singular vectors (shape: [out_features, rank])
- `.B`: Right singular vectors (shape: [rank, in_features])
- `.*bias`: Optional bias terms (if `peft.bias=True`)

### 2.3 Communication Protocol

#### 2.3.1 Uplink (Client → Server)
1. Client trains both A and B
2. Filter parameters (`.A`, `.B`, optional bias)
3. Sort keys lexicographically
4. Pack into `List[NDArrays]`

#### 2.3.2 Downlink (Server → Client)
1. Server slices global factors to client's rank
2. Sort keys lexicographically
3. Pack into `List[NDArrays]`

---

## 3. Safe Extension Policy

### 3.1 General.py Extension Strategy

#### 3.1.1 `set_params` Extension
```python
def set_params(model, parameters, method=None, rank_spec=None, **kwargs):
    """
    Extended to support FlexLoRA's rank adaptation.
    
    Args:
        model: Target model
        parameters: List of parameter arrays
        method: If 'flex_lora', enable rank adaptation
        rank_spec: Dict containing rank information
    """
    if method != "flex_lora":
        # Original behavior for other methods
        return _original_set_params(model, parameters, **kwargs)
    
    # FlexLoRA-specific logic
    return _flex_lora_set_params(model, parameters, rank_spec, **kwargs)
```

#### 3.1.2 `_slice_pad_lora_params` API Contract
```python
def _slice_pad_lora_params(source_tensor, target_shape, rank_spec):
    """
    Slice or pad source tensor to match target shape based on rank_spec.
    
    Args:
        source_tensor: Input tensor to be adapted
        target_shape: Desired output shape
        rank_spec: Dict containing 'param_type' ('A' or 'B') and rank info
        
    Returns:
        Tensor with adapted shape
    """
    # Implementation details...
    pass
```

### 3.2 Safety Guarantees
1. **Backward Compatibility**:
   - Existing code paths remain unchanged when `method != "flex_lora"`
   - No changes to function signatures of public APIs

2. **Determinism**:
   - Same input always produces same output
   - No randomness in slice/pad operations

3. **Error Handling**:
   - Explicit shape validation
   - Clear error messages for mismatched dimensions

---

## 4. Implementation Tasks

### 4.1 Core Components

#### 4.1.1 `mak/utils/general.py`
- [ ] Extend `set_params` with `method` and `rank_spec` parameters
- [ ] Implement `_slice_pad_lora_params` helper
- [ ] Add input validation and error handling

#### 4.1.2 `mak/clients/flex_lora_client.py`
- [ ] Subclass `BaseClient`
- [ ] Implement parameter filtering and sorting
- [ ] Handle rank adaptation in `set_parameters`

#### 4.1.3 `mak/strategies/flex_lora_strategy.py`
- [ ] Implement SVD merge logic
- [ ] Handle rank adaptation during aggregation
- [ ] Slice global factors for downlink

### 4.2 Testing Strategy

#### 4.2.1 Unit Tests
- [ ] `_slice_pad_lora_params` with various input shapes
- [ ] Rank adaptation in both directions (slice and pad)
- [ ] Error cases (invalid shapes, missing keys)

#### 4.2.2 Integration Tests
- [ ] End-to-end training with heterogeneous ranks
- [ ] Round-trip parameter passing
- [ ] Recovery from network errors

---

## 5. Validation Checklist

### 5.1 Protocol Compliance
- [ ] Deterministic Name-Based Sorted List protocol maintained
- [ ] Backward compatibility with FFA-LoRA
- [ ] No modification to base classes

### 5.2 Functional Requirements
- [ ] Support for heterogeneous ranks
- [ ] Correct SVD merge behavior
- [ ] Proper error handling

### 5.3 Performance
- [ ] No significant overhead from rank adaptation
- [ ] Memory usage within expected bounds
- [ ] Training stability across rounds

---

## 6. References
- FlexLoRA paper: https://openreview.net/pdf?id=gkOzoHBXUw
- FlexLoRA code: https://github.com/alibaba/FederatedScope/tree/FlexLoRA
- Flower documentation: https://flower.dev/docs/
- PyTorch SVD: https://pytorch.org/docs/stable/generated/torch.linalg.svd.html