from bayes_opt import BayesianOptimization, util
from bayes_opt.target_space import TargetSpace
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
import numpy as np
import os
import json
import argparse
import subprocess
import time

import run_longbench_BO
from scipy.optimize import NonlinearConstraint

# --- Configuration ---
NUM_LAYERS = 32
NUM_MODIFIABLE_LAYERS = 16  # We're now modifying 16 layers
NUM_REPLACED_LAYERS = 8 # we choose 8 layers to be replaced from last 16 layers
NUM_REUSE_LAYERS = 8 # we choose 8 layers to replace those 8 layers from first 16 layers
# MIN_REUSE_LAYERS = 8 - NUM_MODIFIABLE_LAYERS

# --- Helper Functions ---
def layers_to_remap(layers_config):
    """Converts a list of layer configurations to a remap dictionary.

    Args:
        layers_config: A dictionary representing the target layer for each modifiable layer, 
                       split into two parts: replaced layers and reuse layers.

    Returns:
        A dictionary where keys are original layer indices and values are the 
        target layer indices (after remapping).
    """
    remap = {}

    # Identity mapping for the first 16 layers
    for i in range(NUM_LAYERS - NUM_MODIFIABLE_LAYERS):
        remap[i] = i

    # Mapping for the modifiable layers
    replaced_layers = layers_config['replaced_layers']
    reuse_layers = layers_config['reuse_layers']
    
    replaced_indices = sorted(replaced_layers.keys())

    
    for i, target_index in enumerate(replaced_indices):
        remap[target_index] = reuse_layers[i]

    return remap

def constraints(layer_config):
    """
    Checks if a layer configuration is valid based on the constraints.
    """
    remap = layers_to_remap(layer_config)
    
    # Check for out-of-bounds targets and targets greater than the current layer
    for i, target in remap.items():
        if target < 0 or target >= NUM_LAYERS:
            return False
        if target > i:
            return False

    # Check that the replaced layers are unique and within the correct range
    replaced_layers = layer_config['replaced_layers']
    if len(set(replaced_layers.values())) != NUM_REPLACED_LAYERS:
        return False
    for layer_index in replaced_layers.values():
        if layer_index < NUM_LAYERS - NUM_MODIFIABLE_LAYERS or layer_index >= NUM_LAYERS:
            return False
    
    reuse_layers = layer_config['reuse_layers']
    for layer_index in reuse_layers:
        if layer_index < 0 or layer_index >= NUM_LAYERS - NUM_MODIFIABLE_LAYERS:
            return False

    return True

def constraint_function(layer_config):
    """
    Constraint function for Bayesian Optimization.
    """
    if not constraints(layer_config):
        return -1000
    return 0

constra = NonlinearConstraint(constraint_function, -1, np.inf)

def calculate_memory_usage(remap):
    """Calculates the memory usage for a given remapping.

    Args:
        remap: A dictionary representing the remapping.

    Returns:
        The memory usage (number of unique layers used).
    """
    return len(set(remap.values()))

def run_eval_sh():
    print('run_eval_sh')
    run_longbench_BO.longbench()
    #     # Run the script
    #     completed_process = subprocess.run(
    #         ["bash", "scripts/scripts_longBench/eval.sh"],  # Use "bash" to execute the script
    #         text=True,  # Decode output as text
    #         check=True  # Raise an exception if the script fails
    #     )

def run_metrics_sh():
    print('run_metrics_sh')
    try:
        completed_process = subprocess.run(
            ["bash", "scripts/scripts_longBench/metrics.sh", "results_long_bench/0e9e39f249a16976918f6564b8830bc894c89659_512"],  # Use "bash" to execute the script
            text=True,  # Decode output as text
            check=True  # Raise an exception if the script fails
        )
    except subprocess.CalledProcessError as e:
        # Script failed
        print(f"Error running script: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")

def evaluate_llm_performance(remap):
    with open("current_result.json", "r") as f:
        result = json.load(f)
    return float(result)

# --- Modified Objective Function ---
def objective_function(dic):
    """
    Objective function for Bayesian Optimization.
    """
    # Convert kwargs to a dictionary format for easier handling
    layers_config = {
        'replaced_layers': {int(dic[f'replaced_layer{i}']):int(dic[f'replaced_layer{i}']) for i in range(NUM_REPLACED_LAYERS)},
        'reuse_layers': [int(dic[f'reuse_layer{i}']) for i in range(NUM_REUSE_LAYERS)]
    }

    layer_map_filepath = "layer_map.json"
    with open(layer_map_filepath, "w") as f:
        json.dump(layers_to_remap(layers_config), f)
    
    if not constraints(layers_config):
        return -1000

    # Run the eval and metrics scripts
    run_eval_sh()
    run_metrics_sh()

    # Evaluate performance
    score = evaluate_llm_performance(layers_to_remap(layers_config))

    return score

# --- Bayesian Optimization with Constraint Enforcement ---

# Define the parameter bounds
pbounds = {}
pbounds.update({f'replaced_layer{i}': (NUM_LAYERS - NUM_MODIFIABLE_LAYERS, NUM_LAYERS -1) for i in range(NUM_REPLACED_LAYERS)})
pbounds.update({f'reuse_layer{i}': (0, NUM_LAYERS - NUM_MODIFIABLE_LAYERS - 1) for i in range(NUM_REUSE_LAYERS)})

# Create a custom utility function with constraints
def constrained_utility(x, x_dict, gp, y_max):
    """
    Constrained utility function for Bayesian Optimization.
    """
    layer_config = {
        'replaced_layers': {int(x_dict[f'replaced_layer{i}']):int(x_dict[f'replaced_layer{i}']) for i in range(NUM_REPLACED_LAYERS)},
        'reuse_layers': [int(x_dict[f'reuse_layer{i}']) for i in range(NUM_REUSE_LAYERS)]
    }
    if not constraints(layer_config):
        return -1000  # Return 0 if constraints are not met

    return util.ucb(x, gp, 0.1)  # Use UCB as the base utility function

# Create a BayesianOptimization object
def dummy_target_func(**kwargs):
  """
  A dummy target function that is not actually used.
  """
  return 0

# --- Function to Generate Valid Initial Points ---
def generate_valid_initial_points(num_points, num_layers, num_modifiable_layers, num_replaced_layers, num_reuse_layers):
    """
    Generates valid initial points for the Bayesian Optimization.

    Args:
        num_points: The number of initial points to generate.
        num_layers: The total number of layers.
        num_modifiable_layers: The number of layers that can be modified.
        num_replaced_layers: The number of layers to be replaced.
        num_reuse_layers: The number of layers to be reused.

    Returns:
        A list of valid initial points (configurations for modifiable layers).
    """
    valid_points = []
    while len(valid_points) < num_points:
        # Randomly select layers to be replaced from the last 16 layers
        replaced_layers = np.random.choice(
            range(num_layers - num_modifiable_layers, num_layers), size=num_replaced_layers, replace=False
        )
        
        # Randomly select layers to reuse from the first 16 layers
        reuse_layers = np.random.choice(
            range(num_layers - num_modifiable_layers), size=num_reuse_layers, replace=True
        )
        
        layer_config = {
            'replaced_layers': {layer: layer for layer in replaced_layers},
            'reuse_layers': list(reuse_layers)
        }
        
        if constraints(layer_config):
            valid_points.append(layer_config)

    return valid_points

# Create a BayesianOptimization object
optimizer = BayesianOptimization(
    f=None,
    constraint = constra,
    pbounds=pbounds,
    verbose=2,
)
logger = JSONLogger(path="./logs_.json")

optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
# Create a TargetSpace object
space = TargetSpace(target_func=dummy_target_func, pbounds=pbounds, random_state=1)

# --- Generate and Register Valid Initial Points ---
initial_points = generate_valid_initial_points(num_points=8, num_layers=NUM_LAYERS, num_modifiable_layers=NUM_MODIFIABLE_LAYERS, num_replaced_layers=NUM_REPLACED_LAYERS, num_reuse_layers=NUM_REUSE_LAYERS)
for layer_config in initial_points:
    # y_probe = objective_function({f'layer{i}': layer_config[i] for i in range(NUM_MODIFIABLE_LAYERS)})
    # optimizer.register(params={f'layer{i}': layer_config[i] for i in range(NUM_MODIFIABLE_LAYERS)}, target=y_probe, constraint_value=constraint_function({f'layer{i}': layer_config[i] for i in range(NUM_MODIFIABLE_LAYERS)}))
    
    y_probe = objective_function({f'replaced_layer{i}': layer_config['replaced_layers'][sorted(list(layer_config['replaced_layers'].keys()))[i]] for i in range(NUM_REPLACED_LAYERS)} | {f'reuse_layer{i}': layer_config['reuse_layers'][i] for i in range(NUM_REUSE_LAYERS)})
    optimizer.register(params={f'replaced_layer{i}': layer_config['replaced_layers'][sorted(list(layer_config['replaced_layers'].keys()))[i]] for i in range(NUM_REPLACED_LAYERS)} | {f'reuse_layer{i}': layer_config['reuse_layers'][i] for i in range(NUM_REUSE_LAYERS)}, target=y_probe, constraint_value=constraint_function(layer_config))
    
# --- Optimization Process ---
counter = 0
while counter < 20:
    x_probe = optimizer.space.random_sample()
    variable_names = optimizer.space.keys
    x_dict = dict(zip(variable_names, x_probe))

    # Ensure the suggestion is a numpy array of the correct shape
    x_probe = x_probe.reshape(1, -1)
    
    # Predict the utility of the new point
    y_probe = constrained_utility(x_probe, x_dict, optimizer._gp, optimizer._space.target.max())
    
    # Format x_probe to be a dictionary for the objective function
    x_probe_dict = x_dict
    
    # Evaluate the objective function for the new point
    y_actual = objective_function(x_probe_dict)
    if y_actual > -100 :
        counter+=1
    
    # Register the new point with the observed value
    optimizer.register(params=x_probe_dict, target=y_actual, constraint_value=constraint_function(
        {'replaced_layers': {int(x_dict[f'replaced_layer{i}']):int(x_dict[f'replaced_layer{i}']) for i in range(NUM_REPLACED_LAYERS)},
         'reuse_layers': [int(x_dict[f'reuse_layer{i}']) for i in range(NUM_REUSE_LAYERS)]}
    ))

# --- Results ---
print("Best remapping configuration found:")
best_remap_config = {
    'replaced_layers': {int(optimizer.max['params'][f'replaced_layer{i}']): int(optimizer.max['params'][f'replaced_layer{i}']) for i in range(NUM_REPLACED_LAYERS)},
    'reuse_layers': [int(optimizer.max['params'][f'reuse_layer{i}']) for i in range(NUM_REUSE_LAYERS)]
}
best_remap = layers_to_remap(best_remap_config)
print(best_remap)
print("Best performance score:", optimizer.max['target'])