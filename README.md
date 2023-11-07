# Master-Thesis

## Inference with proposed prompts

For a quick start, you can directly run inference script with my provided prompts (proposed in the thesis).

For inference, run python `inference.py --dataset="gsm8k" --model="code-davinci-002" --method="active_cot" --qes_limit=10 --prompt_path="./inference_prompts/gsm8k_k=10" --random_seed=42 --multipath=1 --temperature=0.7 --api_time_interval=2`.

# Project Name

## Inference with Proposed Prompts

Welcome to the [Your Project Name] repository! This guide will walk you through running inference with the provided prompts from the thesis.

## Quick Start

To get started quickly, follow these steps:

1. **Clone the Repository**: If you haven't already, clone this GitHub repository to your local machine.

   ```bash
   git clone https://github.com/yourusername/your-repo.git

2. **Install Poetry Environment**:
If you haven't already, make sure to install Poetry by running the following command. Poetry is used to manage dependencies and create a virtual environment for your project.

   ```bash
   pip install poetry

Then, use `poetry install` to install the dependencies that are already defined in pyproject.toml file.

```bash
   poetry install
