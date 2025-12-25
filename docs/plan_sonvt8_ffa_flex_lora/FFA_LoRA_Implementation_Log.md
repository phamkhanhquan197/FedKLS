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
- **CRITICAL FIX (Dimension Mismatch, SVD init)**: changed SVD init of A from `U[:, :rank]` to **right singular vectors** `Vh[:rank, :]` to ensure A has shape `[rank, In]`.

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

### 3) `mak/clients/ffa_lora_client.py` (NEW)
- New client class `FFALoRAClient(BaseClient)`.

Downlink set_parameters:
- If incoming parameter length equals full LoRA length -> load **A+B**, then freeze all `.A`.
- If incoming parameter length equals B-only length -> load **B only**, keep A as-is, then freeze all `.A`.

Uplink get_parameters:
- If `config['round']==1` or `config['server_round']==1`: return full `[A1,B1,A2,B2,...]`.
- Else: return B-only `[B1,B2,...]`.

### 4) `mak/strategies/ffa_lora_strategy.py` (NEW)
- New strategy class `FFALoRAStrategy(FedAvg)`.

Downlink (configure_fit):
- `server_round == 1`: send **FULL** parameters (A+B).
- `server_round > 1`: send **B-only**.

Uplink aggregation (aggregate_fit):
- Performs **weighted average only on B**.
- Keeps global A unchanged by reconstructing a new FULL list where A slots unchanged and B slots replaced by aggregated B.

### 5) `mak/servers/ffa_lora_server.py` (NEW)
- Skeleton `FFALoRAServer(ServerSaveData)` for logging compatibility.

### 6) `mak/clients/__init__.py`
- Registered `FFALoRAClient` in `get_client_class` under strategy name `ffalora`.

### 7) `mak/strategies/__init__.py`
- Exported `FFALoRAStrategy` as `FFALoRA` so `get_strategy` can instantiate it by name.

### 8) `main.py`
- Added `ffa_lora` into the branch that applies `apply_svd_to_model` for LoRA-enabled runs.
  - Now `lora_method in ["pissa", "milora", "middle", "lora", "ffa_lora"]` triggers adapter injection.

---

## Remaining verification checklist (manual)
- [ ] Run with `--method ffa_lora --strategy FFALoRA --enabled True`
- [ ] Confirm in logs that A params have `requires_grad=False` on clients.
- [ ] Confirm round 1 sends full params, round 2+ sends B-only.
- [ ] Confirm server aggregates only B (A constant across rounds).

---

## Files changed/added
- Modified:
  - `config.yaml`
  - `mak/utils/helper.py`
  - `mak/clients/__init__.py`
  - `mak/strategies/__init__.py`
  - `main.py`
- Added:
  - `mak/clients/ffa_lora_client.py`
  - `mak/strategies/ffa_lora_strategy.py`
  - `mak/servers/ffa_lora_server.py`
  - `FFA_LoRA_Implementation_Log.md`
