import flwr as fl
from logging import INFO
from flwr.common.logger import log
from datasets.utils.logging import disable_progress_bar
import os
from mak.utils.helper import get_device_and_resources
from mak.utils.helper import gen_dir_outfile_server, get_model, get_strategy, get_server, save_simulation_history,get_dataset, get_size_weights
from mak.utils.pytorch_transformations import TransformationPipeline, TextTransformationPipeline
from mak.clients import get_client_fn
from mak.utils.dataset_info import dataset_info
from mak.utils.helper import get_config, set_seed, parse_args, apply_svd_to_model
from mak.utils.helper import compute_client_distributions, compute_KL_divergence
import copy

def main():
    # Parse arguments and configs
    args = parse_args()
    config_sim = get_config(args.config)
    seed = args.seed if args.seed else config_sim.get('common', {}).get('seed', 42)
    strategy = args.strategy if args.strategy else config_sim.get('server', {}).get('strategy', '')
    dirichlet_alpha = args.dirichlet_alpha if args.dirichlet_alpha else config_sim.get('common', {}).get('dirichlet_alpha', '')
    config_sim['common']['seed'] = seed
    config_sim['server']['strategy'] = strategy
    config_sim['common']['dirichlet_alpha'] = dirichlet_alpha
    
    set_seed(seed=config_sim['common']['seed'])
    
    fds, centralized_testset = get_dataset(config_sim=config_sim)

    if config_sim['server']['strategy'] == 'FedLaw':
        size_weights = get_size_weights(federated_dataset=fds,num_clients=config_sim['server']['num_clients']) #for fedlaw only
    else:
        size_weights = []
    
    dataset_name = fds._dataset_name
    model_name = config_sim['common']['model']
    shape = dataset_info[dataset_name]["input_shape"]

    # Check if the dataset is a text dataset and use the appropriate transformation pipeline
    if dataset_name in ['SetFit/20_newsgroups', 'legacy-datasets/banking77', 'fancyzhx/dbpedia_14'] or model_name in ['distilbert-base-uncased', 'microsoft/deberta-v3-base', 'llama2-7b']:
        # For text datasets, we need to use a different transformation pipeline
        transformation_pipeline = TextTransformationPipeline(dataset_name=dataset_name, model_name=model_name)
    else: 
        # For image datasets, we can use the existing transformation pipeline
        transformation_pipeline = TransformationPipeline(dataset_name=dataset_name)
    # Get the transformations for train and test data
    apply_transforms, apply_transforms_test = transformation_pipeline.get_transformations()

    device, ray_init_args, client_res = get_device_and_resources(config_sim=config_sim)
    out_file_path, saved_models_path = gen_dir_outfile_server(config=config_sim)
    
    if config_sim["common"]["save_log"]:
        fl.common.logger.configure(identifier="FLNCLAB", filename=os.path.join(saved_models_path,'log.txt'))

    #Base model
    base_model = get_model(config_sim,shape = shape)
    log(INFO, f"Base model structure: {base_model}")
    for name, tensor in base_model.state_dict().items():
        log(INFO, f"{name}: shape {tuple(tensor.shape)}")  
    log(INFO, f"=>>>>>>>>>>>>>>>>>Number of layers: {len(base_model.state_dict())}")

    log(INFO,"*"*75)
    #Prepare the server and client models based on the strategy and method
    lora_enabled = config_sim['peft']['enabled']
    lora_method =  config_sim['peft']['method']
    
    #Apply SVD if LoRA is enabled
    if lora_enabled:
        #Decide the client model based on the LoRA method
        if lora_method == "fedkls":
            #Compute client distributions and kl_norm values
            client_distributions = compute_client_distributions(config = config_sim, dataset=fds,num_clients=config_sim['server']['num_clients'])
            kl_normalized_per_client = compute_KL_divergence(client_distributions, num_classes=dataset_info[dataset_name]["num_classes"])
            print(f"KL Normalized per client: {kl_normalized_per_client}")
            print("Mean KL Normalized per client: ", sum(kl_normalized_per_client.values())/len(kl_normalized_per_client))
            for cid, kl_norm_val in kl_normalized_per_client.items():
                log(INFO, f"Client {cid}: Normalized KL Divergence = {kl_norm_val:.4f}")

            #Pre-apply SVD to each client's model using their KL_norm value
            client_models = {}
            for cid in range(config_sim['server']['num_clients']):
                kl_norm = kl_normalized_per_client[cid]
                client_model = copy.deepcopy(base_model)  # Create a copy for each client
                client_model = apply_svd_to_model(model=client_model, config=config_sim, kl_norm=kl_norm, client_id=cid)
                client_models[cid] = client_model
            log(INFO, "FedKLS method enabled: Applied SVD to client models with client-specific kl_norm.")

            log(INFO, "Applying SVD to create svd model for server...")
            # Create a deep copy of base_model to avoid modifying it
            model_for_svd = copy.deepcopy(base_model)
            svd_model = apply_svd_to_model(model=model_for_svd, config=config_sim, kl_norm= sum(kl_normalized_per_client.values())/len(kl_normalized_per_client))
            log(INFO, f"Model after SVD: {svd_model}")
            for name, tensor in svd_model.state_dict().items():
                log(INFO, f"{name}: shape {tuple(tensor.shape)}")  
            log(INFO, f"=>>>>>>>>>>>>>>>>>Number of layers: {len(svd_model.state_dict())}")
            #Server always needs the SVD-adapted model when LoRA is enabled
            server_model = svd_model
            
        elif lora_method in ["pissa", "milora", "middle", "lora"]:
            log(INFO, "Applying SVD to create svd model for server...")
            # Create a deep copy of base_model to avoid modifying it
            model_for_svd = copy.deepcopy(base_model)
            svd_model = apply_svd_to_model(model=model_for_svd, config=config_sim)
            log(INFO, f"Model after SVD: {svd_model}")
            for name, tensor in svd_model.state_dict().items():
                log(INFO, f"{name}: shape {tuple(tensor.shape)}")  
            log(INFO, f"=>>>>>>>>>>>>>>>>>Number of layers: {len(svd_model.state_dict())}")
            #Server always needs the SVD-adapted model when LoRA is enabled
            server_model = svd_model
            
            _ = compute_client_distributions(config = config_sim, dataset=fds,num_clients=config_sim['server']['num_clients'])
            client_model = svd_model
            log(INFO, f"=>>>>> Method {lora_method.upper()}: Sending svd_model to clients.")
        else:
            log(INFO, f"Unknown LoRA method {lora_method}. Defaulting to base_model for clients.")
            import sys
            sys.exit(0)  # Exit if an unknown LoRA method is specified

    else: # If LoRA is not enabled, use the base model for both server and clients (Full fine-tuning)
        log(INFO, "=>>>>> LoRA is not enabled: Using base_model for both server and clients.")
        log(INFO, "=>>>>> Full fine-tuning training!!!")
        _ = compute_client_distributions(config = config_sim, dataset=fds,num_clients=config_sim['server']['num_clients'])
        server_model = base_model
        client_model = base_model

    try:
        dir_alpha = fds._partitioners['train']._alpha[0]
    except (AttributeError):
        dir_alpha = "NA"

    log(INFO,f" =>>>>> Dataset : {dataset_name}") 
    log(INFO,f" =>>>>> Model : {base_model._get_name()} Device : {device}")
    log(INFO,f" =>>>>> Partitoner : {config_sim['common']['data_type']} Alpha : {dir_alpha}")
    log(INFO,f" =>>>>> Ray init args : {ray_init_args} Client Res : {client_res}")

    strategy = get_strategy(
        config=config_sim,
        test_data=centralized_testset,
        save_model_dir=saved_models_path,
        out_file_path= out_file_path,
        device=device,
        apply_transforms_test=apply_transforms_test,
        size_weights=size_weights,
        model=server_model)
    
    server = get_server(
        strategy = strategy,
        client_manager=fl.server.client_manager.SimpleClientManager(),
        out_file_path=out_file_path,
        target_acc=config_sim['common']['target_acc'],
        num_train_thread=config_sim['common']['num_train_thread'],
        num_test_thread=config_sim['common']['num_test_thread'],
    )
    
    log(INFO,f" =>>>>> Using Strategy : {strategy.__class__} Server : {server.__class__}")
    #Update client_fn to pass kl_norm along with the model
    def client_fn_with_models(cid):
        cid = int(cid)
        if lora_method == "fedkls":
            model = client_models[cid]
            kl_norm = kl_normalized_per_client[cid]
        else:
            model = client_model
            kl_norm = None
        return get_client_fn(
            config_sim=config_sim,
            dataset=fds,
            model=model,
            device=device,
            apply_transforms=apply_transforms,
            save_dir=saved_models_path,
            kl_norm_dict=kl_normalized_per_client if lora_method == "fedkls" else None,  # Pass precomputed kl_norms
        )(cid)

    
    hist = fl.simulation.start_simulation(
        client_fn=client_fn_with_models,
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
    main()