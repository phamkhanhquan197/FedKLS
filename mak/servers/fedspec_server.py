from mak.servers.custom_server import ServerSaveData, fit_clients, evaluate_clients
from flwr.common.logger import log
from logging import DEBUG, INFO
import numpy as np
import copy
import timeit
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters

class FedSpecServer(ServerSaveData):
    """FedSpec Server skeleton (logging and orchestration in ServerSaveData)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache expert pools: {(layer_idx, rank): experts}
        self.expert_pool_cache = {}
        # Store previous routing scores for stability
        self.prev_scores = {}

    def build_expert_pool(self, W, rank):
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        k = len(S)

        # Number of exerts depened on rank and layer size
        num_experts = max(1, k // rank)     
        expert_size = rank

        experts = []
        for i in range(num_experts):
            start = i * expert_size
            end = (i + 1) * expert_size if i < num_experts - 1 else k
            experts.append({"U": U[:, start:end], "S": S[start:end], "Vt": Vt[start:end, :]})
        return experts

    def get_expert_pool(self, W, layer_idx, rank):

        key = (layer_idx, rank)
        if key not in self.expert_pool_cache:
            # print(f"[SVD] Build pool: layer={layer_idx}, rank={rank}")
            self.expert_pool_cache[key] = self.build_expert_pool(W, rank)
        # else:
            # print(f"[CACHE] Reuse pool: layer={layer_idx}, rank={rank}")

        return self.expert_pool_cache[key]
    
    def compute_scores(self, experts, kl_norm=None, prev_scores=None):

        # Energy of each expert (spectral importance)
        energies = np.array([np.sum(e["S"]**2) for e in experts]) + 1e-8

        if kl_norm is not None:
            if prev_scores is not None and len(prev_scores) == len(experts):
                 # Smooth transition between previous routing and new spectrum
                scores = (1 - kl_norm) * prev_scores + kl_norm * energies
            else:
                scores = energies
        else:
            scores = energies

        # ---- Stable softmax (avoid overflow) ----
        scores = scores - np.max(scores)
        weights = np.exp(scores) / np.sum(np.exp(scores))

        return weights
    # def combine_experts(self, selected_experts):

    #     U_all, S_all, Vt_all = [], [], []

    #     for e in selected_experts:
    #         U_all.append(e["U"])
    #         S_all.append(e["S"])
    #         Vt_all.append(e["Vt"])

    #     U_cat = np.concatenate(U_all, axis=1)
    #     S_cat = np.concatenate(S_all)
    #     Vt_cat = np.concatenate(Vt_all, axis=0)

    #     sqrt_S = np.sqrt(S_cat)

    #     A = U_cat * sqrt_S
    #     B = sqrt_S[:, None] * Vt_cat

    #     return A, B

    # ============
    # def combine_experts(self, selected_experts, weights=None):
    #     """
    #     Combine experts WITHOUT increasing rank.
    #     Each expert: U (d, r), S (r,), Vt (r, d)
    #     """

    #     k = len(selected_experts)

    #     # ---- Step 1: normalize weights ----
    #     if weights is None:
    #         weights = np.ones(k) / k
    #     else:
    #         weights = np.array(weights)
    #         weights = weights / np.sum(weights)

    #     # ---- Step 2: accumulate in A/B space ----
    #     A_sum = None
    #     B_sum = None

    #     for w, e in zip(weights, selected_experts):
    #         U = e["U"]          # (d, r)
    #         S = e["S"]          # (r,)
    #         Vt = e["Vt"]        # (r, d)

    #         sqrt_S = np.sqrt(S)

    #         A = U * sqrt_S              # (d, r)
    #         B = sqrt_S[:, None] * Vt    # (r, d)

    #         if A_sum is None:
    #             A_sum = w * A
    #             B_sum = w * B
    #         else:
    #             A_sum += w * A
    #             B_sum += w * B

    #     return A_sum, B_sum

    # A_list = []
    # B_list = []

    # for w, e in zip(weights, selected_experts):
    #     U = e["U"]
    #     S = e["S"]
    #     Vt = e["Vt"]

    #     sqrt_S = np.sqrt(S)

    #     A = U * sqrt_S[np.newaxis, :]
    #     B = sqrt_S[:, np.newaxis] * Vt

    #     # weight correctly
    #     A_list.append(np.sqrt(w) * A)
    #     B_list.append(np.sqrt(w) * B)

    # A_cat = np.concatenate(A_list, axis=1)  # (d, k*r)
    # B_cat = np.concatenate(B_list, axis=0)  # (k*r, d)

    # return A_cat, B_cat

    def combine_experts(self, selected_experts, weights=None, rank=None):
        """
        Combine selected experts via concatenation to preserve all spectral information.
        
        Strategy: Concatenate U, S, Vt from all selected experts to keep full spectral structure.
        Then extract top `rank` components to match target rank.

        Args:
            selected_experts: list of {"U", "S", "Vt"}
            weights: importance weights (applied to S values)
            rank: target rank for output

        Returns:
            A_combined, B_combined  (d × rank, rank × d)
        """

        k = len(selected_experts)

        # ---- Step 1: normalize weights (safe) ----
        if weights is None:
            weights = np.ones(k) / k
        else:
            weights = np.array(weights, dtype=np.float64)
            weights = np.clip(weights, 1e-8, None)   # prevent collapse
            weights = weights / np.sum(weights)

        # ---- Step 2: Concatenate expert spectra to preserve all information ----
        U_list = []
        S_list = []
        Vt_list = []

        for w, e in zip(weights, selected_experts):
            U = e["U"]        # (d, rank)
            S = e["S"]        # (rank,)
            Vt = e["Vt"]      # (rank, d)

            # Weight the singular values (importance of this expert)
            S_weighted = w * S

            U_list.append(U)
            S_list.append(S_weighted)
            Vt_list.append(Vt)

        # Concatenate: U_cat (d, k*rank), S_cat (k*rank,), Vt_cat (k*rank, d)
        U_cat = np.concatenate(U_list, axis=1)              # (d, k*rank)
        S_cat = np.concatenate(S_list, axis=0)              # (k*rank,)
        Vt_cat = np.concatenate(Vt_list, axis=0)            # (k*rank, d)

        # ---- Step 3: Extract top `rank` components to match target rank ----
        # Sort by singular value magnitude (descending)
        top_idx = np.argsort(S_cat)[-rank:][::-1]  # top rank indices
        
        U_selected = U_cat[:, top_idx]             # (d, rank)
        S_selected = S_cat[top_idx]                # (rank,)
        Vt_selected = Vt_cat[top_idx, :]           # (rank, d)

        # ---- Step 4: Convert to A, B factors ----
        sqrt_S = np.sqrt(np.maximum(S_selected, 1e-8))  # numerical stability
        
        A_combined = U_selected * sqrt_S[np.newaxis, :]   # (d, rank)
        B_combined = sqrt_S[:, np.newaxis] * Vt_selected  # (rank, d)

        return A_combined, B_combined




    def svd_moe(self, W_global, param_names, rank_dict, kl_dict, client_id): #Repeat each client

        rank = rank_dict.get(client_id, None)
        kl_norm = kl_dict.get(client_id, None)

        if rank is None:
            raise ValueError(f"Rank not found for client {client_id} in rank_dict")
        if kl_norm is None:
            raise ValueError(f"KL norm not found for client {client_id} in kl_dict")
        
        A_list, B_list = [], []
        
        for layer_idx, (name, W) in enumerate(zip(param_names, W_global)):
            if name.startswith("distilbert") and "lin" in name and ".bias" not in name:
                # ---- Step 1: Expert Pool ----
                experts = self.get_expert_pool(W, layer_idx, rank)
                k = len(experts)

                

                # ---- Step 2: KL-based spectral slicing ----
                region_size = max(1, min(rank, k)) # Number of experts to select based on rank constraint
                start = int(kl_norm * (k - region_size))  # Start index based on KL norm
                end = start + region_size  # End index for slicing experts
                region_experts = experts[start:end]

                # ---- Step 3: Previous scores ----
                prev = self.prev_scores.get(client_id, {}).get(layer_idx, None)

                # ---- Step 4: MoE scoring within region ----
                weights_all  = self.compute_scores(region_experts, kl_norm=kl_norm, prev_scores=prev)

                # Select top-k experts based on weights
                top_k = min(max(2, len(region_experts) // 2), len(region_experts))
                selected_idx = np.argsort(weights_all)[-top_k:]
                selected_experts  = [region_experts[i] for i in selected_idx]

                # ---- NEW: rank + energy aware weighting ----
                energies = np.array([np.sum(e["S"]**2) for e in selected_experts])
                # normalize energy (avoid scale explosion)
                energies = energies / (np.sum(energies) + 1e-12)

                # get corresponding scores
                selected_scores = weights_all[selected_idx]

                # combine score + energy
                selected_weights = energies * selected_scores
                # normalize
                selected_weights = selected_weights / (np.sum(selected_weights) + 1e-12)

                # Save scores for next round
                if client_id not in self.prev_scores:
                    self.prev_scores[client_id] = {}
                self.prev_scores[client_id][layer_idx] = selected_scores

                # ---- Step 8: Combine experts (NEW SAFE VERSION) ----
                A, B = self.combine_experts(
                    selected_experts,
                    weights=selected_weights,
                    rank=rank
                )

                # # ---- Energy retained computation ----

                # # Total energy from ALL experts
                # S_all = np.concatenate([e["S"] for e in experts])
                # energy_total = np.sum(S_all ** 2)

                # # Energy from SELECTED experts
                # S_selected = np.concatenate([e["S"] for e in selected_experts])
                # energy_selected = np.sum(S_selected ** 2)

                # energy_ratio = energy_selected / (energy_total + 1e-12)

                # print(f"[Layer {layer_idx}] Energy retained: {energy_ratio:.4f}")


                # W_recon = U @ np.diag(S) @ Vt
                # error = np.linalg.norm(W - W_recon) / np.linalg.norm(W)
                # print(f"[Layer {layer_idx}] Recon error:", error)

                A_list.append(A)
                B_list.append(B)

        return A_list, B_list

    def fit_round(self, server_round, num_rounds, timeout):
        """Perform a single round of federated averaging."""
        log(INFO, "======================================Round %s/%s======================================", server_round, num_rounds)

        # Step 0: Sample clients
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        new_client_instructions = []
        total_download = 0.0

        # =========================
        # Build personalized models
        # =========================
        for client, ins in client_instructions:
            client_id = int(client.cid)
            rank_dict = ins.config.get("rank_dict", {})
            kl_dict = ins.config.get("kl_dict", {})
            param_names = ins.config.get("param_names", [])

            rank = rank_dict.get(client_id, None)
            log(INFO, f"Training client {client.cid} with rank {rank}")

            # Clone parameters
            W_global = parameters_to_ndarrays(ins.parameters)
            W_global = [w.copy() for w in W_global]

            # ---- Step 1: SVD-MoE ----
            A_list, B_list = self.svd_moe(W_global, param_names, rank_dict, kl_dict, client_id)

            # ---- Step 2: Build personalized weights ----
            personalized_W = []
            svd_idx = 0

            for name, W in zip(param_names, W_global):
                if name.startswith("distilbert") and "lin" in name and ".bias" not in name:
                    A = A_list[svd_idx].copy()
                    B = B_list[svd_idx].copy()

                    personalized_W.append(A)
                    personalized_W.append(B)
                    svd_idx += 1

                elif server_round == 1:
                    personalized_W.append(W)

                elif name.startswith("distilbert") and "lin" in name and ".bias" in name:
                    personalized_W.append(W)

            # ---- Step 3: Convert ----
            personalized_params = ndarrays_to_parameters(personalized_W)

            # ---- Step 4: Attach to client ----
            new_ins = type(ins)(
                parameters=personalized_params,
                config=ins.config
            )
            new_client_instructions.append((client, new_ins))

        # Replace instructions
        client_instructions = new_client_instructions

        if not client_instructions:
            log(INFO, "Start training: no clients selected, cancel")
            return None

        # =========================
        # DOWNLOAD (Server → Client)
        # =========================
        for client, ins in client_instructions:
            client_id = int(client.cid)
            rank = ins.config.get("rank_dict", {}).get(client_id, None)

            client_params = ins.parameters
            client_download = sum(len(t) for t in client_params.tensors) / 1e9
            client_download_mb = client_download * 1024

            # ✅ Log per-client model size (DOWNLOAD)
            log(
                INFO,
                f"[Download] Client {client.cid} | Rank: {rank} | "
                f"Size: {client_download:.6f} GB ({client_download_mb:.2f} MB)"
            )

            # tracking
            self.comm_tracker.per_client[client.cid]["download"] += client_download
            total_download += client_download

        # round-level download
        self.comm_tracker.log_round(
            server_round=server_round,
            upload=0,
            download=total_download,
        )

        log(
            INFO,
            f"[Download Summary] Total: {total_download:.4f} GB "
            f"({total_download*1024:.2f} MB), Avg/client: {(total_download/len(client_instructions))*1024:.2f} MB"
        )

        log(
            DEBUG,
            "Start training: sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # =========================
        # TRAINING
        # =========================
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            num_threads=self.num_train_thread,
        )


        # =========================
        # CLIENT METRICS LOGGING
        # =========================
        for _, fit_res in results:
            client_id = fit_res.metrics["client_id"]
            train_samples = fit_res.num_examples
            class_dist = fit_res.metrics["class_distribution"]
            num_class = len(class_dist)
            rank = fit_res.metrics["rank"]
            kl_norm = fit_res.metrics["kl_norm"]

            log(
                INFO,
                f"Client {client_id} | Training Samples: {train_samples} | "
                f"Classes ({num_class}): {class_dist} | "
                f"Next Rank: {rank} | KL: {kl_norm}"
            )

        log(
            DEBUG,
            "Server evaluation with %s results and %s failures",
            len(results),
            len(failures),
        )
        
        # =========================
        # UPLOAD (Client → Server)
        # =========================
        total_upload = 0.0

        for client, fit_res in results:
            client_id = int(client.cid)

            client_upload = sum(len(t) for t in fit_res.parameters.tensors) / 1e9
            client_upload_mb = client_upload * 1024

            # ✅ Log per-client upload size
            log(
                INFO,
                f"[Upload] Client {client.cid} |" 
                f"Size: {client_upload:.6f} GB ({client_upload_mb:.2f} MB)"
            )

            self.comm_tracker.per_client[client.cid]["upload"] += client_upload
            total_upload += client_upload

        # round-level upload
        self.comm_tracker.per_round[server_round]["upload"] = total_upload
        self.comm_tracker.per_round[server_round]["download"] = total_download
        self.comm_tracker.total_upload += total_upload

        log(
            INFO,
            f"[Round {server_round} Communication] "
            f"Upload: {total_upload:.4f} GB ({total_upload*1024:.2f} MB), "
            f"Download: {total_download:.4f} GB ({total_download*1024:.2f} MB), "
            f"Total: {(total_upload + total_download):.4f} GB "
            f"({(total_upload + total_download)*1024:.2f} MB)"
        )

        # =========================
        # AGGREGATION
        # =========================
        parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(
            server_round, results, failures
        )

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def evaluate_round(self, server_round, timeout, curr_round_start_time):
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        log(INFO, "**************************************************")
        if not client_instructions:
            log(INFO, "Start evaluating: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "Start evaluating: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        new_client_instructions = []

        for client, ins in client_instructions:
            client_id = int(client.cid)
            rank_dict = ins.config.get("next_rank_dict", None)
            kl_dict = ins.config.get("kl_dict", None)
            param_names = ins.config.get("param_names", None)

            log(INFO, f"Testing client {client.cid} with rank {rank_dict.get(client_id)}")

            # Clone parameters (avoid shared references)
            W_global = parameters_to_ndarrays(ins.parameters)
            W_global = [w.copy() for w in W_global]

            # ---- Step 1: SVD-MoE ----
            A_list, B_list = self.svd_moe(W_global, param_names, rank_dict, kl_dict, client_id)

            # ---- Step 2: Build heterogeneous parameter list ----
            personalized_W = []

            svd_idx = 0

            for name, W in zip(param_names, W_global):

                if name.startswith("distilbert") and "lin" in name and ".bias" not in name:
                    A = A_list[svd_idx].copy()
                    B = B_list[svd_idx].copy()

                    # Keep A and B (NO reconstruction)
                    personalized_W.append(A)
                    personalized_W.append(B)
    
                    svd_idx += 1
                elif name.startswith("distilbert") and "lin" in name and ".bias" in name:
                    personalized_W.append(W)

            # ---- Step 3: Convert ----
            personalized_params = ndarrays_to_parameters(personalized_W)

            # ---- Step 4: Rebuild FitIns (critical) ----
            new_ins = type(ins)(
                parameters=personalized_params,
                config=ins.config
            )
            new_client_instructions.append((client, new_ins))

        # ✅ Replace safely
        client_instructions = new_client_instructions

        #Check the shape of the parameters sent to clients
        # for client, ins in client_instructions:
        #     param_shapes = [p.shape for p in parameters_to_ndarrays(ins.parameters)]
        #     log(INFO, f"Client {client.cid} received parameters with shapes: {param_shapes}")


        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            num_threads=self.num_test_thread,  # Number of concurrent threads
        )

        log(
            DEBUG,
            "Client evaluation with %s results and %s failures",
            len(results),
            len(failures),
        )

        for i in range(len(results)):
            client_id = results[i][1].metrics["client_id"]
            val_samples = results[i][1].num_examples
            acc = results[i][1].metrics["accuracy"]
            f1 = results[i][1].metrics["f1_score"]
            loss = results[i][1].loss
            num_class =  len(results[i][1].metrics["class_distribution"])
            class_dist = results[i][1].metrics["class_distribution"]

            log(
                INFO,
                f"Client {client_id} | Validation Samples: {val_samples} | "
                f"Classes ({num_class}): {class_dist} | "
                f"Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}"
            )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        log(
            INFO,
            "Local results (avg): (round %s: loss: %s, metric: %s, processing time: %s)",
            server_round,
            loss_aggregated,
            metrics_aggregated,
            timeit.default_timer() - curr_round_start_time,
        )
        return loss_aggregated, metrics_aggregated, (results, failures)
