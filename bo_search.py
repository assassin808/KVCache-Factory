from bayes_opt import BayesianOptimization, util  # Import util
from bayes_opt.target_space import TargetSpace
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
import numpy as np
import os
import json
import argparse
import subprocess
import time



# --- Configuration ---
NUM_LAYERS = 32  # Example: Total number of layers in your LLM
MIN_REUSE_LAYERS = 8  # Minimum number of layers that must be reused

# --- Helper Functions ---
def layers_to_remap(layers_config):
  """Converts a list of layer configurations to a remap dictionary.

  Args:
    layers_config: A list representing the target layer for each original layer.

  Returns:
    A dictionary where keys are original layer indices and values are the 
    target layer indices (after remapping).
  """
  remap = {}
  for i, target in enumerate(layers_config):
    remap[i] = int(target)  # Ensure integer values
  return remap

def is_valid_remap(remap):
    """Checks if a remapping configuration is valid.

    Args:
        remap: A dictionary representing the remapping.

    Returns:
        True if the remapping is valid, False otherwise.
    """
    used_targets = set()
    for i, target in enumerate(remap.values()):
      if target < 0 or target >= NUM_LAYERS:
        return False
      if target > i:
        return False  # Target layer cannot be after the current layer
      if target in used_targets and target != i:
        return False # only former layers can replace later layers
      used_targets.add(target)

    # Check for minimum layer reuse (at least 8 distinct target layers)
    if len(set(remap.values())) > NUM_LAYERS - MIN_REUSE_LAYERS:
        return False
    return True

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
    try:
        # Run the script
        completed_process = subprocess.run(
            ["bash", "scripts/scripts_longBench/eval.sh"],  # Use "bash" to execute the script
            text=True,  # Decode output as text
            check=True  # Raise an exception if the script fails
        )
    except subprocess.CalledProcessError as e:
        # Script failed
        print(f"Error running script: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")

def run_metrics_sh():
    print('run_metrics_sh')
    try:
        completed_process = subprocess.run(
            ["bash", "scripts/scripts_longBench/metrics.sh", "results_long_bench/0e9e39f249a16976918f6564b8830bc894c89659_2048"],  # Use "bash" to execute the script
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

def constraints(layer_config):
    """
    Checks if a layer configuration is valid based on the constraints.
    """
    remap = layers_to_remap(layer_config)
    
    used_targets = set()
    for i, target in enumerate(remap.values()):
        if target < 0 or target >= NUM_LAYERS:
            return False
        if target > i:
            return False
        if target in used_targets:
            return False
        if target != i:
            used_targets.add(i)

    if len(set(remap.values())) > NUM_LAYERS - MIN_REUSE_LAYERS:
        return False

    return True

# --- Modified Objective Function ---
def objective_function(**kwargs):
    """
    Objective function for Bayesian Optimization.
    """
    # Convert kwargs to a list format for easier handling
    layers_config = [int(kwargs[f'layer{i}']) for i in range(NUM_LAYERS)]

    layer_map_filepath = "layer_map.json"
    with open(layer_map_filepath, "w") as f:
        json.dump(layers_to_remap(layers_config), f)

    # Run the eval and metrics scripts
    # run_eval_sh()
    # run_metrics_sh()

    # Evaluate performance
    # score = evaluate_llm_performance(layers_to_remap(layers_config))
    import random
    score = random.random()
    return score

# --- Bayesian Optimization with Constraint Enforcement ---

# Define the parameter bounds
pbounds = {f'layer{i}': (0, i) for i in range(NUM_LAYERS)}

# Create a custom utility function with constraints
def constrained_utility(x, gp, y_max):
    """
    Constrained utility function for Bayesian Optimization.
    """
    layer_config = [int(val) for val in x[0]]
    if not constraints(layer_config):
        return 0  # Return 0 if constraints are not met

    return util.ucb(x, gp, 0.1)  # Use UCB as the base utility function

# Create a BayesianOptimization object
def dummy_target_func(**kwargs):
  """
  A dummy target function that is not actually used.
  """
  return 0

# --- Function to Generate Valid Initial Points ---
def generate_valid_initial_points(num_points, num_layers, min_reuse_layers):
    """
    Generates valid initial points for the Bayesian Optimization.

    Args:
        num_points: The number of initial points to generate.
        num_layers: The total number of layers.
        min_reuse_layers: The minimum number of layers to reuse.

    Returns:
        A list of valid initial points (layer configurations).
    """
    valid_points = []
    while len(valid_points) < num_points:
        layer_config = list(range(num_layers))  # Start with a default configuration

        # Determine the number of layers to reuse (randomly choose between min_reuse_layers and num_layers)
        num_layers_to_reuse = np.random.randint(min_reuse_layers, num_layers)
        # Randomly select layers to be replaced
        layers_to_replace = np.random.choice(
            range(num_layers), size=num_layers - num_layers_to_reuse, replace=False
        )

        # Replace the selected layers with valid alternatives
        for layer_index in layers_to_replace:
            valid_targets = list(range(layer_index + 1))
            layer_config[layer_index] = np.random.choice(valid_targets)

        # Check if the generated configuration is valid
        # print(len(layer_config))
        if constraints(layer_config):
            valid_points.append(layer_config)

    return valid_points

# Create a BayesianOptimization object
optimizer = BayesianOptimization(
    f=None,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)
logger = JSONLogger(path="./logs.json")

optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
# Create a TargetSpace object
space = TargetSpace(target_func=dummy_target_func, pbounds=pbounds, random_state=1)

# --- Generate and Register Valid Initial Points ---
initial_points = generate_valid_initial_points(num_points=8, num_layers=NUM_LAYERS, min_reuse_layers=MIN_REUSE_LAYERS)
for layer_config in initial_points:
    x_probe = np.array(layer_config)
    y_probe = objective_function(**{f'layer{i}': layer_config[i] for i in range(NUM_LAYERS)})
    optimizer.register(params=x_probe, target=y_probe)

# --- Optimization Process ---


for _ in range(3):
    x_probe = optimizer.space.random_sample()
    
    # Ensure the suggestion is a numpy array of the correct shape
    x_probe = x_probe.reshape(1, -1)
    
    # Predict the utility of the new point
    y_probe = constrained_utility(x_probe, optimizer._gp, optimizer._space.target.max())
    
    # Format x_probe to be a dictionary for the objective function
    x_probe_dict = {f'layer{i}': int(x_probe[0][i]) for i in range(NUM_LAYERS)}
    
    # Evaluate the objective function for the new point
    y_actual = objective_function(**x_probe_dict)
    
    # Register the new point with the observed value
    optimizer.register(params=x_probe[0], target=y_actual)

# --- Results ---
print("Best remapping configuration found:")
best_remap = layers_to_remap([int(optimizer.max['params'][f'layer{i}']) for i in range(NUM_LAYERS)])
print(best_remap)
print("Best performance score:", optimizer.max['target'])
