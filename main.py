from mak.utils import get_config, set_seed
config_sim = get_config('./config.yaml')
set_seed(seed=config_sim['common']['seed'])
from mak.client import get_client_fn
import flwr as fl
from logging import INFO, DEBUG
from flwr.common.logger import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from datasets.utils.logging import disable_progress_bar
from mak.custom_server import ServerSaveData
from mak.utils import gen_dir_outfile_server, get_model, get_strategy,save_simulation_history,get_dataset
import torch
from mak.pytorch_transformations import get_transformations
from mak.utils import get_device_and_resources

def main(config_sim):
    out_file_path, saved_models_path = gen_dir_outfile_server(config=config_sim)

    fds, centralized_testset = get_dataset(config_sim=config_sim)
    dataset_name = fds._dataset_name

    apply_transforms = get_transformations(dataset_name = dataset_name)

    model = get_model(config_sim)

    device, ray_init_args, client_res = get_device_and_resources(config_sim=config_sim)
    
    #log all config here
    try:
        dir_alpha = fds._partitioners['train']._alpha[0]
    except (AttributeError):
        dir_alpha = "NA"
    log(INFO,f" =>>>>> Model : {model._get_name()} Device : {device}")
    log(INFO,f" =>>>>> Dataset : {dataset_name} Partitoner : {str(fds._partitioners['train']).split('.')[-1]} Alpha : {dir_alpha}")
    log(INFO,f" =>>>>> Ray init args : {ray_init_args} Client Res : {client_res}")

    strategy = get_strategy(config=config_sim,test_data=centralized_testset,save_model_dir=saved_models_path,out_file_path= out_file_path,device=device,apply_transforms=apply_transforms)
    server = ServerSaveData(
        strategy=strategy, client_manager=fl.server.client_manager.SimpleClientManager(),out_file_path=out_file_path,target_acc=config_sim['common']['target_acc'])
    log(INFO,f" =>>>>> Using Strategy : {strategy.__class__} Server : {server.__class__}")

    hist = fl.simulation.start_simulation(
        client_fn=get_client_fn(model=model,dataset=fds,device=device,apply_transforms=apply_transforms),
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