# FedKLS: Federated KL-Driven Low-rank SVD Adaptation in Non-IID Data Distributions

<!-- ![FedKLS radar chart](placeholder-for-logo.png) -->


FedKLS is an intuitive and powerful yet simple-to-use research framework for KL-guided, SVD-based low-rank personalization in federated learning, built on top of [Flower](https://flower.ai/) and [FedEasy](https://fedeasy.readthedocs.io/) frameworks. It enables scalable, communication-efficient experiments for non-IID federated scenarios featuring advanced parameter-efficient adaptation (e.g., LoRA, PiSSA, MiLoRA, and FedKLS).

This repository provides an easy-to-use Federated Learning framework based on Flower and PyTorch. It's designed to simplify the process of setting up and running federated learning experiments.

> **Note**: This code has been written and tested on Ubuntu and should work seamlessly on any Linux-based distribution. Windows users may need to adjust some steps.

## 📋 Prerequisites

- Git
- Python 3.11.x
- pip (Python package installer)

## 🚀 Getting Started

Follow these steps to set up and use this repository:

### 1. Clone the Repository

```bash
git clone https://github.com/phamkhanhquan197/FedKLS.git
```

### 2. Create and activate a `venv` environment
```bash
python3 -m venv FedKLS
cd FedKLS/
source bin/activate
```

### 3. Upgrade `pip` and install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 4. Configure Your Experiment
Edit `config.yaml` to define your desired dataset, model, client distribution, FL strategy, etc.

### 5. Run FedKLS Simulation
```bash
python3 main.py
```
If you are using an Nvidia GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py
```
All outputs (logs, metrics) are saved in the output/ directory.

### 6. Analyze Results using `compute_results.py`
Install dependencies (if not installed):
```bash
pip install pandas
```
Edit `compute_results.py` to set your CSV log and your target F1-score, for example:
```bash
threshold=0.9
csv_path = os.path.join(os.getcwd(), 'output', '2025-09-15', 'FedAWA', 'dirichlet_niid', 'pissa_20news_50', 'FedAWA_SetFit_20_newsgroups_dirichlet_niid_32_0.005_1.csv')
```
Run the script
```bash
python3 compute_results.py
```

## Description about  [`config.yaml`](/config.yaml) file
The `config.yaml` file is a configuration file for this framework that trains a Federated Learning model.

The configuration file is divided into three sections: `common`, `server`, and `client`.

### Common Section
The `common` section contains the common configurations used in this framework. 

- `data_type`: This field specifies the data distribution type used in the training process. Currently supported data distributions are [`iid`,`dirichlet_niid`] . Detailed explination can be found [here](./docs/data_distribution.md)
- `dataset`: This field specifies the dataset used in the training process. Currently supported data distributions are [`SetFit/20_newsgroups`, `legacy-datasets/banking77`, `fancyzhx/dbpedia_14`]. Detailed explination can be found [here](./docs/datasets.md)
- `dirichlet_alpha`: This field is used when `data_type` is set to `dirichlet_niid`. It specifies the Dirichlet concentration parameter.
- `target_acc`: This field specifies the target accuracy that the model needs to achieve. It can take any value greater than `0`.
- `model`: This field specifies the model architecture used in the training process. Currently implemented models are [ `distilbert-base-uncased`, `bert-base-uncased`, `Qwen/Qwen1.5-0.5B`]. Detailed explination can be found [here](./docs/models.md)
- `optimizer`: This field specifies the optimizer used in the training process. It could be either `sgd` or `adam`.
- `seed`: This field fixes the seed for reproducibility.
- `multi_node`: set to `True` for distributed runs.
- `save_log`: `True/False` to store logs in output directory.
- `num_train_thread`, `num_test_thread` : Control thread usage for I/O and dataloader efficiency.

### Server Section
The `server` section contains the configurations for the server that coordinates the Federated Learning process.

- `num_rounds`: This field specifies the maximum number of rounds for the training process.
- `address`: This field specifies the IP address of the server.
- `fraction_fit`: This field specifies the fraction of participating clients used for training in each round.
- `min_fit_clients`: This field specifies the minimum number of participating clients required for training in each round.
- `num_clients`: Total number of clients participating in training.
- `fraction_evaluate`: This field specifies the fraction of participating clients used for evaluation in each round.
- `min_evaluate_clients`: This field specifies that at least this many clients are required for evaluation.
- `strategy`: This field specifies the strategy used for Federated Learning. Currently supported strategies are [`FedAWA`, `FedAvg`, `FedSVD`, `FedKLSVD`, `Scaffold`, `FedNova`, `FedProx`, ...]. Detailed explanation can be found [here](./docs/strategies.md)

### Client Section
The `client` section contains the configurations for the clients participating in the Federated Learning process.

- `epochs`: This field specifies the number of epochs for each client's training process.
- `batch_size`: This field specifies the batch size for each client's training process.
- `lr`: This field specifies the learning rate for each client's training process.
- `save_train_res`: This field specifies whether to save the training results. It could be either `true` or `false`.
If `save_train_res` is set to `true`, all the output data, like accuracy, loss, and time of each round, would be saved in the `out` directory.
- `total_cpus`: No. of CPU cores that are assigned for all simulations.
- `total_gpus`:  No. of GPUs assigned for the whole simulation.
- `gpu`: True or False, Use GPU for training or not. Default to False
- `num_cpus`: No. of CPU cores assigned for each actor. Default to 1
- `num_gpus`: Fraction of GPU assigned to each actor. (num_cpus and num_gpus can only be used in simulation mode if `simulation` is set to `True`) For more details on this, please refer to https://flower.dev/docs/framework/how-to-run-simulations.html and https://docs.ray.io/en/latest/ray-core/scheduling/resources.html

### PEFT Section
- `enabled` : `True` = Use Parameter-Efficient Fine-Tuning (PEFT); `False` = Full Fine-Tuning (FFT).
- `rank`: Low-rank dimension for adapters.
- `alpha`: LoRA/adapter alpha parameter (typically same as rank).
- `method`: Choose among 'lora', 'pissa', 'milora', 'middle', 'fedkls' (Do not set `method` = 'fedkls' when running FFT)

### FedSVD Config Section
When `server.strategy: FedSVD`, you can control the aggregation behavior with `fedsvd_config` in `config.yaml`:

- `mode`: `"fedavg"` (aggregate LoRA A+B) or `"ffa"` (aggregate only LoRA B).
- `send_deltas`: if `True`, treat client updates as deltas and apply them to the round-start global params.
- `agg_flora`, `agg_fedex`: advanced variants from the upstream FedSVD repo (require delta semantics; keep `False` unless the client is ported for it).

### Dynamic Data Section
- `enabled`: `True` to enable dynamic dataset updates; `False` to use static datasets (default).
- `mode`: Choose between `"incremental"` (dataset size increases monotonically) or `"reset"` (full dataset repartitioning at milestones).
- `round_step`: Frequency of dataset updates (e.g., `1` = every round, `10` = every 10 rounds).
- `start_fraction`: For incremental mode, initial data fraction per client (default: `0.3`). Dataset grows linearly to 1.0 by final round.

### FedAWA Config Section
- `server_valid_ratio`: Fraction for server validation.
- `server_epochs`: Local epochs on the server side.
- `server_optimizer`: Optimizer for server, e.g. adam.
- `server_lr`: Learning rate on the server side.
- `gamma`: Weight for FL regularization.
- `reg_distance`: Type of regularization distance, e.g. cosine.









