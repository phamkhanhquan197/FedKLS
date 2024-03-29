from mak.utils import get_config, set_seed
config_sim = get_config('./config.yaml')
set_seed(seed=config_sim['common']['seed'])
from mak.client import get_client_fn
import flwr as fl
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from datasets.utils.logging import disable_progress_bar
from mak.custom_server import ServerSaveData
from mak.utils import gen_dir_outfile_server, get_model, get_strategy,save_simulation_history
import torch
def main(config_sim):
    out_file_path, saved_models_path = gen_dir_outfile_server(config=config_sim)
    fds = FederatedDataset(dataset=config_sim['common']['dataset'], partitioners={"train": config_sim['server']['num_clients']})
    centralized_testset = fds.load_split("test")

    model = get_model(config_sim)
    device = "cpu"

    if config_sim['client']['num_cpus'] and (config_sim['client']['num_gpus'] > 0.0 and config_sim['client']['num_gpus'] <= 1.0):
        client_res = {'num_cpus': config_sim['client']['num_cpus'], 'num_gpus' : config_sim['client']['num_gpus']}
    else:
        client_res = {'num_cpus': config_sim['client']['num_cpus'], 'num_gpus' : 0.0}

    if config_sim['client']['gpu']:
        ray_init_args = {'num_cpus': config_sim['client']['total_cpus'], 'num_gpus' : config_sim['client']['total_cpus']}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        ray_init_args = {'num_cpus': config_sim['client']['total_cpus'], 'num_gpus' : 0}
        client_res = {'num_cpus': config_sim['client']['num_cpus'], 'num_gpus' : 0.0}
    
    print("Using device:", device)
    
    strategy = get_strategy(config=config_sim,test_data=centralized_testset,save_model_dir=saved_models_path,device=device)
    server = ServerSaveData(
        strategy=strategy, client_manager=fl.server.client_manager.SimpleClientManager(),out_file_path=out_file_path,target_acc=config_sim['common']['target_acc'])
    print(f"Using Strategy : {strategy.__class__} and server : {server.__class__}")
    hist = fl.simulation.start_simulation(
        client_fn=get_client_fn(model=model,dataset=fds,device=device),
        num_clients=config_sim['server']['num_clients'],
        client_resources=client_res,
        config=fl.server.ServerConfig(num_rounds=config_sim['server']['num_rounds']),
        strategy=strategy,
        server=server,
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
        ray_init_args = ray_init_args,
    )

    simu_data_file_path = out_file_path.replace('.csv','_metrics.csv')
    save_simulation_history(hist=hist,path = simu_data_file_path)

if __name__ == "__main__":
    main(config_sim=config_sim)