# Few-shot chain-of-thought prompting with LLMs

## Python version requirement
The pyproject.toml file specifies that Python versions from 3.9 to 3.9.6, as well as Python versions from 3.9.7 up to, but not including, 4.0 are allowed.

## Reproduce results
When trying to reproduce the results from this work, keep in mind that 
* OpenAI models are non-deterministic, meaning that identical inputs can yield different outputs. Setting temperature to 0 will make the outputs mostly deterministic, but a small amount of variability may remain due to GPU floating point math (Source: https://community.openai.com/t/run-same-query-many-times-different-results/140588)
* Use the estimated uncertainties that are exported into a JSON file to prevent varied uncertainty scores, as using a temperature of 0.7 might yield different outcomes, as well the exported question embeddings.

## Setup

To get started quickly, follow these steps:

1. **Clone the Repository**: If you haven't already, clone this GitHub repository to your local machine and navigate to the cloned directory.

   ```bash
   git clone https://github.com/Lori10/Master-Thesis-Few-Shot-CoT-Prompting-LLM.git

2. **Install Poetry**:
If you haven't already, make sure to install Poetry by running the following command. Poetry is used to manage dependencies and create a virtual environment for your project.

   ```bash
   pip install poetry

3. **Install Poetry Environment**:
Create poetry environment and install the dependencies that are already defined in pyproject.toml file.

   ```bash
   poetry install

4. **Set environment variables**:
In my thesis, I use the AzureOpenAI model in langchain, which is accessible through the UPTIMIZE GPT API, provided by Merck KGaA, acting as a proxy for the Azure OpenAI API. To configure your environment for using the OpenAI model in langchain instead of AzureOpenAI, copy the .env.template file to your project root directory, rename it as .env, edit the newly created .env file and add your OPENAI_API_KEY as an environment variable.


## Inference with generated Prompts/Demos from the thesis experiments

For a quick start, you can directly run the `inference.py` script on the test dataset using the prompts and demos provided in the thesis.

1. **Get demos**: In the `labeled_demos` directory, you can explore prompts and demos generated using various prompting methods, such as Random-CoT, Diverse-CoT, Active-CoT, etc., as mentioned in the thesis. Each method has its own subdirectory. Within these subdirectories, you will find directories named with the date and time when the demos/prompts were generated. This organization is useful as different hyperparameters were used at each time

   ```bash
   python inference.py --dataset="gsm8k" --data_path="../datasets/original/gsm8k/test.jsonl" --dir_prompts="labeled_demos/random/2023_08_29_22_30_28/demos" --model_id="gpt-3.5-turbo-0613" --random_seed=1 --method="cot" --temperature=0.0 --output_dir="inference_results" --dataset_size_limit=0

## Demos/Prompt Generation with Different Prompting Methods and Inference

1. **Demo Generation**: Each of the following python file is dedicated to a particular prompting method and generates the desired demos or prompts. 
To create in-context demos using specific prompting methods, you can use the following Python files:

- For Random-CoT: `generate_random.py`
- For Active-CoT: `generate_active.py`
- For Diverse-CoT: `generate_diverse.py`
- For Retrieval-CoT: `generate_demo_run_inference_retrieval.py`
- For Diverse-Active-KMeans-CoT: `generate_demo_diverse_active_cot_kmeans_plusplus.py`
- For Diverse-Active-KMeansPlusPlus-CoT: `generate_demo_run_inference_diverse_active_cot_kmeans_plusplus_retrieval.py`
- For Diverse-Active-KMeansPlusPlus-Retrieval-CoT: `generate_demo_run_inference_diverse_active_cot_kmeans_plusplus_retrieval.py`

To run the Diverse-Active-KMeansPlusPlus-CoT method, use the "generate_demo_run_inference_diverse_active_cot_kmeans_plusplus_retrieval.py" file and set the retrieval argument to False. When setting retrieval to True, you run the prompting method for Diverse-Active-KMeansPlusPlus-Retrieval-CoT, as this adds the retrieval stage after running the Diverse-Active-KMeansPlusPlus-CoT method.

2. **Inference**: After generating demos, you can run the Language Model (LLM) on inference using the generated demos by the prompting method. Only for Retrieval-CoT and Diverse-Active-KMeansPlusPlus-Retrieval-CoT, the demo generation and inference are integrated into the same file, "generate_demo_run_inference_retrieval.py" and "generate_demo_run_inference_diverse_active_cot_kmeans_plusplus_retrieval.py", respectively, as the prompt is different for each test question.




