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
from bayes_opt import UtilityFunction
from bayes_opt.util import load_logs
import os  # Import the 'os' module

import run_longbench_BO
from scipy.optimize import NonlinearConstraint

utility_function = UtilityFunction(kind="ucb", kappa=0.1, xi=0.0)

# --- Configuration ---
NUM_LAYERS = 32
REUSE_LAYER = 2  # Index of the layer to be reused
REPLACED_LAYERS_START = 3  # Start index for layers to be replaced
REPLACED_LAYERS_END = 32  # End index for layers to be replaced (exclusive)
NUM_REPLACED_LAYERS = 8  # Number of layers to be replaced

# --- Helper Functions ---
def layers_to_remap(layers_config):
    """Converts a list of layer configurations to a remap dictionary.

    Args:
        layers_config: A dictionary representing the target layer for each modifiable layer,
                       specifically indicating which layers are replaced.

    Returns:
        A dictionary where keys are original layer indices and values are the
        target layer indices (after remapping).
    """
    remap = {}

    # Identity mapping for layers outside the replaced range
    for i in range(0, REPLACED_LAYERS_START):
        remap[i] = i

    # Mapping for the replaced layers
    replaced_layers = layers_config['replaced_layers']
    for layer_index in replaced_layers:
        remap[layer_index] = REUSE_LAYER

    # Identity mapping for layers after the replaced range
    for i in range(REPLACED_LAYERS_END, NUM_LAYERS):
        remap[i] = i

    return remap

def constraints(layer_config):
    """
    Checks if a layer configuration is valid based on the constraints.
    """
    # Check that the replaced layers are unique and within the correct range
    replaced_layers = layer_config['replaced_layers']
    if len(set(replaced_layers)) != NUM_REPLACED_LAYERS:
        return False
    for layer_index in replaced_layers:
        if layer_index < REPLACED_LAYERS_START or layer_index >= REPLACED_LAYERS_END:
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
        'replaced_layers': [int(dic[f'replaced_layer{i}']) for i in range(NUM_REPLACED_LAYERS)],
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
pbounds.update({f'replaced_layer{i}': (REPLACED_LAYERS_START, REPLACED_LAYERS_END - 1) for i in range(NUM_REPLACED_LAYERS)})

# Create a custom utility function with constraints
def constrained_utility(x, x_dict, gp, y_max):
    """
    Constrained utility function for Bayesian Optimization.
    """
    layer_config = {
        'replaced_layers': [int(x_dict[f'replaced_layer{i}']) for i in range(NUM_REPLACED_LAYERS)],
    }
    if not constraints(layer_config):
        return -1000

    return utility_function.utility(x, gp, y_max)

# Create a BayesianOptimization object
def dummy_target_func(**kwargs):
    """
    A dummy target function that is not actually used.
    """
    return 0

# --- Function to Generate Valid Initial Points ---
def generate_valid_initial_points(num_points, num_layers, num_replaced_layers):
    """
    Generates valid initial points for the Bayesian Optimization.

    Args:
        num_points: The number of initial points to generate.
        num_layers: The total number of layers.
        num_replaced_layers: The number of layers to be replaced.

    Returns:
        A list of valid initial points (configurations for modifiable layers).
    """
    valid_points = []
    while len(valid_points) < num_points:
        # Randomly select layers to be replaced
        replaced_layers = np.random.choice(
            range(REPLACED_LAYERS_START, REPLACED_LAYERS_END), size=num_replaced_layers, replace=False
        )

        layer_config = {
            'replaced_layers': list(replaced_layers),
        }

        if constraints(layer_config):
            valid_points.append(layer_config)

    return valid_points

# Create a BayesianOptimization object
optimizer = BayesianOptimization(
    f=None,
    constraint=constra,
    pbounds=pbounds,
    verbose=2,
)

logs_file = "./logs_prev.json"  # Path to your logs file

if os.path.isfile(logs_file):
    load_logs(optimizer, logs=[logs_file])
    print(f"Loaded logs from {logs_file}. Resuming optimization...")
else:
    print("No previous logs found. Starting from scratch.")

logger = JSONLogger(path="./logs_.json")

optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
# Create a TargetSpace object
space = TargetSpace(target_func=dummy_target_func, pbounds=pbounds, random_state=1)

# --- Generate and Register Valid Initial Points ---
initial_points = generate_valid_initial_points(num_points=8, num_layers=NUM_LAYERS, num_replaced_layers=NUM_REPLACED_LAYERS)
for layer_config in initial_points:
    y_probe = objective_function({f'replaced_layer{i}': layer_config['replaced_layers'][i] for i in range(NUM_REPLACED_LAYERS)})
    optimizer.register(params={f'replaced_layer{i}': layer_config['replaced_layers'][i] for i in range(NUM_REPLACED_LAYERS)}, target=y_probe, constraint_value=constraint_function(layer_config))

# --- Optimization Process ---
counter = 0
while counter < 50:
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
    if y_actual > -100:
        counter += 1

    # Register the new point with the observed value
    optimizer.register(params=x_probe_dict, target=y_actual, constraint_value=constraint_function(
        {'replaced_layers': [int(x_dict[f'replaced_layer{i}']) for i in range(NUM_REPLACED_LAYERS)]}
    ))

# --- Results ---
print("Best remapping configuration found:")
best_remap_config = {
    'replaced_layers': [int(optimizer.max['params'][f'replaced_layer{i}']) for i in range(NUM_REPLACED_LAYERS)],
}
best_remap = layers_to_remap(best_remap_config)
print(best_remap)
print("Best performance score:", optimizer.max['target'])