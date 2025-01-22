from bayes_opt import BayesianOptimization
import numpy as np
import os
import json

import os
import json
import argparse
import numpy as np

import subprocess
import time

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


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
    try:
        # Run the script
        completed_process = subprocess.run(
            ["bash", "scripts/scripts_longBench/eval.sh"],  # Use "bash" to execute the script
            capture_output=True,  # Capture stdout and stderr
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
    try:
        completed_process = subprocess.run(
            ["bash", "scripts/scripts_longBench/metrics.sh [placeholders]"],  # Use "bash" to execute the script
            capture_output=True,  # Capture stdout and stderr
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

# --- Objective Function for Bayesian Optimization ---
def objective_function(**kwargs):
    """
    Objective function for the Bayesian Optimization process.

    Args:
        **kwargs: Keyword arguments representing the layer configuration,
                  e.g., layer0, layer1, ..., layer15.

    Returns:
        The negative LLM performance score (since BayesianOptimization minimizes).
    """
    layers_config = [kwargs[f'layer{i}'] for i in range(NUM_LAYERS)]
    remap = layers_to_remap(layers_config)

    if not is_valid_remap(remap):
        return -1000  # Return a very low score for invalid mappings

    layer_map_filepath = "layer_map.json"  # Or any path you prefer
    with open(layer_map_filepath, "w") as f:
        json.dump(remap, f)

    # Run the eval script
    run_eval_sh()
    # Run the metrics script
    run_metrics_sh()
    # Evaluate the LLM performance
    score = evaluate_llm_performance(remap)
    return score

# --- Bayesian Optimization Setup ---
# Define the bounds for each layer (each layer can map to itself or a previous layer)
pbounds = {}
for i in range(NUM_LAYERS):
    pbounds[f'layer{i}'] = (0, i) 

# Create a BayesianOptimization object
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=1,
    verbose=2  # 2: Prints all results, 1: Prints only when a maximum is observed, 0: No output
)

# --- Optimization Process ---
optimizer.maximize(
    init_points=1,  # Number of initial random tests before Bayesian Optimization starts
    n_iter=1,      # Number of iterations of Bayesian Optimization
)

# --- Results ---
print("Best remapping configuration found:")
best_remap = layers_to_remap([int(optimizer.max['params'][f'layer{i}']) for i in range(NUM_LAYERS)])
print(best_remap)
print("Best performance score:", optimizer.max['target'])