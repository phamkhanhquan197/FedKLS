# Implementation Plan: FFA-LoRA và FlexLoRA Baselines

---

## 📋 Tổng quan

| Câu hỏi | Trả lời |
|---------|---------|
| **Khởi tạo A, B** | A = Kaiming Normal (Gaussian), B = Zero |
| **FFA-LoRA** | A frozen vĩnh viễn, Round 1 gửi A+B, Round 2+ chỉ gửi B |
| **FlexLoRA round đầu** | Client nhận A, B từ Server (theo rank của mình) |
| **FlexLoRA round sau** | Server SVD **MỌI ROUND** → slice → gửi (A_i, B_i) |
| **FlexLoRA rank selection** | 4 Types theo resource budget, Server assign |
| **Thứ tự implement** | FFA-LoRA trước |

> **Lưu ý về Round Number**: 
> - Flower framework đếm round bắt đầu từ **1**
> - Tác giả dùng thuật ngữ "Round 0" để chỉ vòng khởi tạo
> - **Chốt**: Code thực tế sẽ dùng `server_round == 1` của Flower để tương ứng với "Round 0" (Initialization Round) theo ý tưởng của tác giả

---

## 📁 Cấu trúc files mới

```
mak/
├── clients/
│   ├── ffa_lora_client.py      # [NEW] FFA-LoRA client (train B only)
│   └── flex_lora_client.py     # [NEW] FlexLoRA client (nhận A_i, B_i từ server)
├── servers/
│   ├── ffa_lora_server.py      # [NEW] FFA-LoRA server (kế thừa ServerSaveData)
│   └── flex_lora_server.py     # [NEW] FlexLoRA server (kế thừa ServerSaveData)
├── strategies/
│   ├── ffa_lora_strategy.py    # [NEW] FFA-LoRA aggregation (aggregate B only)
│   └── flex_lora_strategy.py   # [NEW] FlexLoRA SVD aggregation + slice per client
└── utils/
    └── helper.py               # [MODIFY] Thêm nhánh ffa_lora, flex_lora trong apply_svd_to_model
```

---

## 🔷 BASELINE 1: FFA-LoRA

### 1.1. Config Changes

```yaml
peft:
  method: ffa_lora
  rank: 8
  alpha: 8

ffa_lora_config:
  seed: 42
  # Options: kaiming | gaussian | orthogonal | svd
  # - kaiming: Kaiming He initialization (default, theo paper FFA-LoRA)
  # - gaussian: Normal distribution với std=0.01
  # - orthogonal: Orthogonal initialization
  # - svd: Dùng Top-r Singular Vectors từ pre-trained weight (Appendix A.8)
  init_method: "kaiming"
```

### 1.2. Helper Functions (`mak/utils/helper.py`)

```python
def apply_svd_to_model(model, config, method, **kwargs):
    """
    Cập nhật hàm apply_svd_to_model để hỗ trợ ffa_lora và flex_lora.
    """
    # ... existing code for pissa, milora, fedkls ...
    
    if method == 'ffa_lora':
        # Khởi tạo A, B (Zero), freeze A
        rank = config.get("peft", {}).get("rank", 8)
        alpha = config.get("peft", {}).get("alpha", 8)
        seed = config.get("ffa_lora_config", {}).get("seed", 42)
        init_method = config.get("ffa_lora_config", {}).get("init_method", "kaiming")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and should_apply_lora(name):
                d_out, d_in = module.weight.shape
                
                # === A Initialization Logic (4 options) ===
                A = torch.empty(d_out, rank)
                
                if init_method == "kaiming":
                    # Kaiming He initialization (default)
                    nn.init.kaiming_normal_(A, mode='fan_out', nonlinearity='relu')
                    
                elif init_method == "orthogonal":
                    # Orthogonal initialization
                    nn.init.orthogonal_(A)
                    
                elif init_method == "gaussian":
                    # Simple Gaussian với std nhỏ
                    nn.init.normal_(A, mean=0, std=0.01)
                    
                elif init_method == "svd":
                    # Paper FFA-LoRA (Appendix A.8): Dùng Top-r Singular Vectors của W0 làm A
                    # W_orig = U @ S @ V^T → A = U[:, :rank]
                    U, S, Vh = torch.linalg.svd(module.weight.data.float(), full_matrices=False)
                    A = U[:, :rank].to(dtype=module.weight.dtype)
                    
                else:
                    raise ValueError(f"Unknown init_method: {init_method}. "
                                     f"Options: kaiming | gaussian | orthogonal | svd")
                
                # === B: Luôn là Zero ===
                B = torch.zeros(rank, d_in)
                
                # === Freeze A ngay từ đầu ===
                A.requires_grad = False
                
                W_res = module.weight.data.clone()
                new_module = SVDAdapter(W_res, A, B, alpha=alpha, rank=rank)
                set_module(model, name, new_module)
        
        return model
    
    elif method == 'flex_lora':
        # Khởi tạo A, B cho phép rank thay đổi linh hoạt
        default_rank = config.get("peft", {}).get("rank", 8)
        alpha = config.get("peft", {}).get("alpha", 8)
        
        # Rank có thể được override bởi client_rank từ config
        rank = kwargs.get("client_rank", default_rank)
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and should_apply_lora(name):
                d_out, d_in = module.weight.shape
                
                # Xác định rank theo layer type nếu cần
                if isinstance(rank, dict):
                    if is_attention_layer(name):
                        r = rank.get("attn", default_rank)
                    else:
                        r = rank.get("ffn", default_rank)
                else:
                    r = rank
                
                A = torch.empty(d_out, r)
                nn.init.kaiming_normal_(A, mode='fan_out')
                B = torch.zeros(r, d_in)
                
                W_res = module.weight.data.clone()
                new_module = SVDAdapter(W_res, A, B, alpha=alpha, rank=r)
                set_module(model, name, new_module)
        
        return model


def is_attention_layer(layer_name):
    """Phân biệt Attention vs FFN layer."""
    attn_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "attention", "self_attn"]
    return any(k in layer_name for k in attn_keywords)
```

### 1.3. FFA-LoRA Client (`mak/clients/ffa_lora_client.py`)

```python
class FFALoRAClient(BaseClient):
    """
    FFA-LoRA Client: Freeze A, Train B only.
    - Round 1: Nhận cả A và B từ server (Flower đếm từ 1)
    - Round 2+: Chỉ nhận B (A giữ nguyên trong buffer)
    """
    
    def __init__(self, client_id, model, trainset, valset, config_sim, device, save_dir):
        super().__init__(client_id, model, trainset, valset, config_sim, device, save_dir)
        self.A_buffer = None  # Lưu A từ Round 1
    
    def set_parameters(self, parameters, config):
        """
        Logic check để xử lý 2 trường hợp:
        - Nhận cả A và B -> Load cả hai, lưu A vào buffer (frozen)
        - Chỉ nhận B -> Load B, giữ nguyên A đang có
        """
        param_dict = self._parameters_to_dict(parameters)
        
        # Check xem có nhận được A không
        has_A = any("lora_A" in key or ".A" in key for key in param_dict.keys())
        
        if has_A:
            # Nhận cả A và B (Round 1)
            self._load_full_lora(param_dict)
            self.A_buffer = self._extract_A_matrices()
            # Freeze A
            for name, param in self.model.named_parameters():
                if "lora_A" in name or ".A" in name:
                    param.requires_grad = False
        else:
            # Chỉ nhận B (Round 2+)
            self._load_B_only(param_dict)
            # A giữ nguyên từ buffer
    
    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        
        # Train (chỉ B được update vì A đã freeze)
        loss = self._train_local()
        
        return self.get_parameters(config), len(self.trainset), {"loss": loss}
    
    def get_parameters(self, config):
        """Chỉ trả về B parameters (không gửi A)."""
        params = []
        for name, param in self.model.state_dict().items():
            if "lora_B" in name or ".B" in name:
                params.append(param.cpu().numpy())
        return params
    
    def _extract_A_matrices(self):
        """Trích xuất và lưu A matrices vào buffer."""
        return {name: param.clone() for name, param in self.model.named_parameters()
                if "lora_A" in name or ".A" in name}
    
    def _load_B_only(self, param_dict):
        """Load chỉ B parameters từ server."""
        current_state = self.model.state_dict()
        for name, param in param_dict.items():
            if "lora_B" in name or ".B" in name:
                current_state[name] = param
        self.model.load_state_dict(current_state, strict=False)
```

### 1.4. FFA-LoRA Strategy (`mak/strategies/ffa_lora_strategy.py`)

```python
class FFALoRAStrategy(FedAvg):
    """
    FFA-LoRA Aggregation:
    - Round 1: Gửi cả A và B (Flower đếm từ 1)
    - Round 2+: Aggregate chỉ B, gửi chỉ B
    """
    
    def __init__(self, model, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.seed = config.get("ffa_lora_config", {}).get("seed", 42)
        self.global_A = None  # Frozen, tạo 1 lần
        self.global_B = None
    
    def initialize_parameters(self, client_manager):
        """Khởi tạo A và B ban đầu."""
        model = apply_svd_to_model(self.base_model, self.config, method='ffa_lora')
        self.global_A = self._extract_A(model)
        self.global_B = self._extract_B(model)
        # Trả về Parameters chứa cả A và B
        return self._combine_A_B()
    
    def configure_fit(self, server_round, parameters, client_manager):
        """
        Cấu hình trước mỗi round.
        - server_round == 1: Gửi cả A + B
        - server_round > 1: Chỉ gửi B
        """
        config = {"server_round": server_round}
        
        if server_round == 1:
            # Round 1: Gửi cả A + B để Client khởi tạo đúng
            params_to_send = self._combine_A_B()
        else:
            # Round 2+: Chỉ đóng gói và gửi B
            params_to_send = self._pack_B_only()
        
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients
        )
        
        return [(client, FitIns(params_to_send, config)) for client in clients]
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate chỉ B."""
        if not results:
            return None, {}
        
        # Thu thập B từ clients
        client_Bs = [fit_res.parameters for _, fit_res in results]
        weights = [fit_res.num_examples for _, fit_res in results]
        
        # Weighted average của B
        self.global_B = self._weighted_average_B(client_Bs, weights)
        
        # Return B (A không đổi)
        return self._pack_B_only(), {}
    
    def _extract_A(self, model):
        return {name: param.clone() for name, param in model.named_parameters()
                if "lora_A" in name or ".A" in name}
    
    def _extract_B(self, model):
        return {name: param.clone() for name, param in model.named_parameters()
                if "lora_B" in name or ".B" in name}
    
    def _combine_A_B(self):
        """Đóng gói cả A và B thành Parameters."""
        combined = {}
        combined.update(self.global_A)
        combined.update(self.global_B)
        return ndarrays_to_parameters([v.numpy() for v in combined.values()])
    
    def _pack_B_only(self):
        """Đóng gói chỉ B thành Parameters."""
        return ndarrays_to_parameters([v.numpy() for v in self.global_B.values()])
```

### 1.5. FFA-LoRA Server (`mak/servers/ffa_lora_server.py`)

```python
from mak.servers.custom_server import ServerSaveData


class FFALoRAServer(ServerSaveData):
    """
    FFA-LoRA Server - Kế thừa từ ServerSaveData.
    
    Có thể thêm custom logic nếu cần, nhưng phần lớn logic
    nằm trong FFALoRAStrategy.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # Có thể override các methods nếu cần custom behavior
    # Ví dụ: logging đặc biệt cho FFA-LoRA, tracking A usage, etc.
```

---

## 🔷 BASELINE 2: FlexLoRA

### 2.1. Config Changes

```yaml
peft:
  method: flex_lora
  rank: 8
  alpha: 8

flex_lora_config:
  distribution: "uniform"     # uniform | heavy_tail_light | heavy_tail_strong | normal
  
  types:
    type1:
      rank: 8
      rank_attn: 8
      rank_ffn: 8
      params_percent: 0.12
    type2:
      rank: 30
      rank_attn: 30
      rank_ffn: 30
      params_percent: 2.46
    type3:
      rank: null              # Mixed
      rank_attn: 30
      rank_ffn: 200
      params_percent: 8.22
    type4:
      rank: 200
      rank_attn: 200
      rank_ffn: 200
      params_percent: 12.22
  
  distributions:
    uniform:
      type1: 0.25
      type2: 0.25
      type3: 0.25
      type4: 0.25
    heavy_tail_light:
      type1: 0.70
      type2: 0.10
      type3: 0.10
      type4: 0.10
    heavy_tail_strong:
      type1: 0.10
      type2: 0.10
      type3: 0.10
      type4: 0.70
    normal:
      type1: 0.10
      type2: 0.35
      type3: 0.35
      type4: 0.20
  
  energy_threshold: 0.95
  max_global_rank: 64
  optimizer: "adamw"
  learning_rate: 1e-4
```

### 2.2. FlexLoRA Client (`mak/clients/flex_lora_client.py`)

```python
class FlexLoRAClient(BaseClient):
    """
    FlexLoRA Client: Nhận A_i, B_i đã được slice sẵn từ Server.
    Client KHÔNG tự resize - Server đã slice theo rank của client.
    """
    
    def __init__(self, client_id, model, trainset, valset, config_sim, device, save_dir):
        super().__init__(client_id, model, trainset, valset, config_sim, device, save_dir)
        self.client_type = None
        self.local_rank = None
    
    def set_parameters(self, parameters, config):
        """Nhận (A_i, B_i) đã được slice sẵn từ Server."""
        self.client_type = config.get("client_type")
        self.local_rank = config.get("local_rank")
        self._load_lora_parameters(parameters)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        optimizer = self._get_optimizer(config)
        loss = self._train_local(optimizer=optimizer)
        
        return self.get_parameters(config), len(self.trainset), {
            "loss": loss,
            "client_type": self.client_type,
            "local_rank": self.local_rank
        }
    
    def _get_optimizer(self, config):
        opt_name = config.get("optimizer", "adamw")
        lr = config.get("learning_rate", 1e-4)
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if opt_name == "adamw":
            return torch.optim.AdamW(params, lr=lr)
        elif opt_name == "adam":
            return torch.optim.Adam(params, lr=lr)
        return torch.optim.SGD(params, lr=lr)
    
    def get_parameters(self, config):
        return {name: p.cpu() for name, p in self.model.state_dict().items()
                if "lora_A" in name or "lora_B" in name}
```

### 2.3. FlexLoRA Strategy (`mak/strategies/flex_lora_strategy.py`) ⚠️ CRITICAL

```python
class FlexLoRAStrategy(FedAvg):
    """
    FlexLoRA Aggregation với SVD Merge.
    
    ⚠️ QUAN TRỌNG: 
    - SVD được thực hiện trong aggregate_fit ở MỌI ROUND
    - KHÔNG có biến cờ _svd_done
    - aggregate_fit trả về parameters rỗng/dummy
    - configure_fit xử lý việc slice và gửi cho từng client
    """
    
    def __init__(self, model, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.flex_config = config.get("flex_lora_config", {})
        
        self.client_types = {}
        self.client_ranks = {}
        
        self.global_A = {}
        self.global_B = {}
        self.global_rank = None
    
    def initialize_parameters(self, client_manager):
        """Khởi tạo và assign client types."""
        num_clients = client_manager.num_available()
        self._assign_client_types(num_clients)
        
        # Khởi tạo global model với default rank
        model = apply_svd_to_model(self.base_model, self.config, method='flex_lora')
        self.global_A = self._extract_A(model)
        self.global_B = self._extract_B(model)
        self.global_rank = self.config.get("peft", {}).get("rank", 8)
        
        return None  # configure_fit sẽ xử lý việc gửi params
    
    def _assign_client_types(self, num_clients):
        """Assign types cho clients dựa trên distribution."""
        distribution = self.flex_config.get("distribution", "uniform")
        probs = self.flex_config["distributions"][distribution]
        
        types = ["type1", "type2", "type3", "type4"]
        type_counts = {t: int(probs[t] * num_clients) for t in types}
        
        client_id = 0
        for t in types:
            for _ in range(type_counts[t]):
                self.client_types[client_id] = t
                self.client_ranks[client_id] = self._get_rank_for_type(t)
                client_id += 1
        
        while client_id < num_clients:
            self.client_types[client_id] = "type2"
            self.client_ranks[client_id] = self._get_rank_for_type("type2")
            client_id += 1
    
    def _get_rank_for_type(self, type_name):
        type_cfg = self.flex_config["types"][type_name]
        return {
            "attn": type_cfg.get("rank_attn", type_cfg.get("rank", 8)),
            "ffn": type_cfg.get("rank_ffn", type_cfg.get("rank", 8))
        }
    
    def configure_fit(self, server_round, parameters, client_manager):
        """
        Chuẩn bị params cho mỗi client - SLICE THEO RANK.
        Được gọi sau aggregate_fit của round trước.
        """
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients
        )
        
        fit_configs = []
        for client in clients:
            cid = int(client.cid)
            client_type = self.client_types.get(cid, "type2")
            client_rank = self.client_ranks.get(cid, {"attn": 30, "ffn": 30})
            
            # SLICE A, B theo rank của client này
            sliced_params = self._slice_global_for_client(cid, client_rank)
            
            config = {
                "server_round": server_round,
                "client_type": client_type,
                "local_rank": client_rank,
                "optimizer": self.flex_config.get("optimizer", "adamw"),
                "learning_rate": self.flex_config.get("learning_rate", 1e-4)
            }
            
            fit_configs.append((client, FitIns(sliced_params, config)))
        
        return fit_configs
    
    def _slice_global_for_client(self, client_id, client_rank):
        """
        Direct Slicing - không cần SVD lại!
        global_A, global_B đã được sắp xếp theo singular values.
        """
        sliced = {}
        
        for layer_name in self.global_A.keys():
            if self._is_attention_layer(layer_name):
                r = client_rank["attn"]
            else:
                r = client_rank["ffn"]
            
            A_global = self.global_A[layer_name]
            B_global = self.global_B[layer_name]
            
            if self.global_rank > r:
                A_sliced = A_global[:, :r]
                B_sliced = B_global[:r, :]
            elif self.global_rank < r:
                d_out, d_in = A_global.shape[0], B_global.shape[1]
                A_sliced = torch.zeros(d_out, r)
                A_sliced[:, :self.global_rank] = A_global
                B_sliced = torch.zeros(r, d_in)
                B_sliced[:self.global_rank, :] = B_global
            else:
                A_sliced = A_global
                B_sliced = B_global
            
            sliced[f"{layer_name}.A"] = A_sliced
            sliced[f"{layer_name}.B"] = B_sliced
        
        return ndarrays_to_parameters([v.numpy() for v in sliced.values()])
    
    def aggregate_fit(self, server_round, results, failures):
        """
        SVD-based aggregation - Thực hiện ở MỌI ROUND.
        
        Quy trình:
        1. Nhận updates (A_i, B_i) từ Clients
        2. Reconstruct W_i = A_i @ B_i (với scale factor)
        3. Tính W_global = weighted_sum(W_i)
        4. SVD: U, S, Vh = svd(W_global)
        5. Lưu global_A, global_B
        6. Trả về parameters rỗng (configure_fit xử lý gửi)
        """
        if not results:
            return None, {}
        
        # Step 1: Thu thập weights
        weights = [fit_res.num_examples for _, fit_res in results]
        total = sum(weights)
        
        # Step 2 & 3: Reconstruct và aggregate W
        W_aggregated = {}
        layer_names = self._get_lora_layer_names()
        
        for layer_name in layer_names:
            W_aggregated[layer_name] = None
            
            for (_, fit_res), w in zip(results, weights):
                params = fit_res.parameters
                A_i = self._extract_layer_A(params, layer_name)
                B_i = self._extract_layer_B(params, layer_name)
                
                # Reconstruct W_i = A_i @ B_i
                # Lưu ý: có thể cần scale factor s = alpha/rank
                W_i = A_i @ B_i
                
                if W_aggregated[layer_name] is None:
                    W_aggregated[layer_name] = (w / total) * W_i
                else:
                    W_aggregated[layer_name] += (w / total) * W_i
        
        # Step 4 & 5: SVD và lưu global A, B
        for layer_name, W_global in W_aggregated.items():
            # TODO: Use torch.svd_lowrank for large models
            U, S, Vh = torch.linalg.svd(W_global, full_matrices=False)
            
            k = self._select_rank_by_energy(S)
            
            # Tạo A_global, B_global từ SVD
            self.global_A[layer_name] = U[:, :k] @ torch.diag(torch.sqrt(S[:k]))
            self.global_B[layer_name] = torch.diag(torch.sqrt(S[:k])) @ Vh[:k, :]
            self.global_rank = k
        
        # Step 6: Trả về parameters rỗng/dummy
        # configure_fit của round sau sẽ xử lý việc slice và gửi
        return ndarrays_to_parameters([]), {}
    
    def _select_rank_by_energy(self, S):
        threshold = self.flex_config.get("energy_threshold", 0.95)
        max_rank = self.flex_config.get("max_global_rank", 64)
        energy = torch.cumsum(S, dim=0) / torch.sum(S)
        k = torch.searchsorted(energy, threshold).item() + 1
        return min(k, max_rank)
    
    def _is_attention_layer(self, layer_name):
        attn_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "attention", "self_attn"]
        return any(k in layer_name for k in attn_keywords)
```

### 2.4. FlexLoRA Server (`mak/servers/flex_lora_server.py`)

```python
from mak.servers.custom_server import ServerSaveData


class FlexLoRAServer(ServerSaveData):
    """
    FlexLoRA Server - Kế thừa từ ServerSaveData.
    
    Có thể thêm custom logic nếu cần:
    - Logging chi tiết về SVD aggregation
    - Tracking rank distribution
    - Monitoring communication cost per client type
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # Override methods nếu cần custom behavior
```

---

## ✅ Validation Checklist

### FFA-LoRA
- [ ] A được khởi tạo bằng Kaiming Normal
- [ ] Round 1: Server gửi cả A và B (Flower đếm từ 1)
- [ ] Round 2+: Server chỉ gửi B
- [ ] Client check: nếu có A → load + freeze, nếu không → chỉ load B
- [ ] `lora_A.requires_grad == False`

### FlexLoRA
- [ ] **SVD được thực hiện MỌI ROUND** trong aggregate_fit
- [ ] **KHÔNG có biến cờ** `_svd_done`
- [ ] Server assign client types theo distribution config
- [ ] Server slice A, B TRƯỚC KHI gửi (trong configure_fit)
- [ ] aggregate_fit trả về parameters rỗng
- [ ] 4 distribution strategies hoạt động đúng

---

## 📝 Implementation Order

### Phase 1: FFA-LoRA
1. [ ] `helper.py`: Thêm nhánh `if method == 'ffa_lora'` trong `apply_svd_to_model`
2. [ ] `ffa_lora_server.py`: Skeleton kế thừa ServerSaveData
3. [ ] `ffa_lora_client.py`: Logic check A+B vs B only
4. [ ] `ffa_lora_strategy.py`: Round 1 vs Round 2+ logic
5. [ ] Test & verify

### Phase 2: FlexLoRA
1. [ ] `helper.py`: Thêm nhánh `if method == 'flex_lora'`
2. [ ] `flex_lora_server.py`: Skeleton kế thừa ServerSaveData
3. [ ] `flex_lora_strategy.py`: `_assign_client_types()`
4. [ ] `flex_lora_strategy.py`: `aggregate_fit()` với SVD MỌI ROUND
5. [ ] `flex_lora_strategy.py`: `_slice_global_for_client()` (Direct Slicing)
6. [ ] `flex_lora_client.py`: Nhận sliced params
7. [ ] Test & verify

---