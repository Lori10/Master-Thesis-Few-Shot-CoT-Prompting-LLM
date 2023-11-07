# Few-shot prompting with LLMs

## Python version requirement
The pyproject.toml file specifies that Python versions from 3.9 to 3.9.6, as well as Python versions from 3.9.7 up to, but not including, 4.0 are allowed.

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
Install the dependencies that are already defined in pyproject.toml file.

   ```bash
   poetry install

4. **Set environment variables**:
In my thesis, I use the Azure OpenAI model accessible through the UPTIMIZE GPT API, provided by Merck KGaA, acting as a proxy for the Azure OpenAI API. To configure your environment for using the OpenAI model in LangChain, copy the .env.template file to your project root directory, rename it as .env, edit the newly created .env file and add your OPENAI_API_KEY as an environment variable.


## Inference with Proposed Prompts/Demos

For a quick start, you can directly run the `inference.py` script on the test dataset using the prompts and demos provided in the thesis.

1. **Get demos**: In the `labeled_demos` directory, you can explore prompts and demos generated using various prompting methods, such as Random-CoT, Diverse-CoT, Active-CoT, etc., as mentioned in the thesis. Each method has its own subdirectory. Within these subdirectories, you will find directories named with the date and time when the demos/prompts were generated. This organization is useful as different hyperparameters were used at each time

   ```bash
   python inference.py --dataset="gsm8k" --data_path="../datasets/original/gsm8k/test.jsonl" --dir_prompts="labeled_demos/random/2023_08_29_22_30_28/demos" --model_id="gpt-3.5-turbo-0613" --random_seed=1 --method="cot" --temperature=0.0 --output_dir="inference_results" --dataset_size_limit=0

## Generating Demos/Prompts with Different Prompting Methods

Each of the following python file is dedicated to a particular prompting method and generates the desired demos or prompts. 
To create in-context demos using specific prompting methods, you can use the following Python files:

- For Random-CoT: `generate_random.py`
- For Active-CoT: `generate_active.py`
- For Diverse-CoT: `generate_diverse.py`
- For Retrieval-CoT: `generate_demo_run_inference_retrieval.py`
- For Diverse-Active-KMeans-CoT: `generate_demo_diverse_active_cot_kmeans_plusplus.py`
- For Diverse-Active-KMeansPlusPlus-CoT: `generate_demo_run_inference_diverse_active_cot_kmeans_plusplus_retrieval`
- For Diverse-Active-KMeansPlusPlus-Retrieval-CoT: `generate_demo_run_inference_diverse_active_cot_kmeans_plusplus_retrieval`

To run the Diverse-Active-KMeansPlusPlus-CoT method, use the generate_demo_run_inference_diverse_active_cot_kmeans_plusplus_retrieval.py file and set the retrieval argument to False. When setting retrieval to True, you run the prompting method for Diverse-Active-KMeansPlusPlus-Retrieval-CoT, as this adds the retrieval stage after running the Diverse-Active-KMeansPlusPlus-CoT method.



