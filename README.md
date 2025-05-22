# 3D Key Reuse: Memory-Efficient Long-Context LLM Inference

This repository contains the codebase for the paper "3D Key Reuse: Memory-Efficient Long-Context LLMs Inference Across Layers, Heads, and Tokens".

The code is functional but undergoing organization. A refined version will be released soon.

## Getting Started

### Prerequisites

Install the necessary dependencies:

```sh
pip install -r requirements.txt
```

### Running the Demo

To evaluate inference performance on LongBenchmark, execute the following script:

```sh
bash scripts/scripts_longBench/eval.sh minicache 256 0.5 hotpotqa
```

**Parameters:**

*   `minicache`: The name of our method.
*   `256`: Cache size.
*   `0.5`: Reuse ratio.
*   `hotpotqa`: Dataset name (e.g., `hotpotqa`, `narrativeqa`, `qasper`).

You can modify these parameters to experiment with different configurations. Ensure the model path is correctly set within the script.

## Model Support

*   **`main` branch:** Supports LLaMA models.
*   **`final_mistral` branch:** Contains code for Mistral models. This branch will be merged into `main` soon.