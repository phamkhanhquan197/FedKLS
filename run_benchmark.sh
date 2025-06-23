#!/bin/bash

CONFIG_FILE="./config.yaml"

# Define the list of algorithms
algorithms=(FedAvg)

# Define the list of seed values
seeds=(1 2 3)

alphas=(0.01 0.1 1.0)

datasets=(SetFit/20_newsgroups legacy-datasets/banking77 fancyzhx/dbpedia_14)

method=(lora pissa milora fedkls)

# Loop through each algorithm
for alpha in "${alphas[@]}"
do
    for algo in "${algorithms[@]}"
    do
        # Loop through each seed value
        for seed in "${seeds[@]}"
        do
            # Run the main.py script with the current algorithm and seed
            echo "Running main.py with algorithm: $algo and seed: $seed dir alpha : $alpha"
            CUDA_VISIBLE_DEVICES=0 python main.py --config="$CONFIG_FILE" --seed="$seed" --strategy="$algo" --dirichlet_alpha="$alpha"
        done
    done
done