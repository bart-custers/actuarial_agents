# User guide

## 1. Installing the environment and packages

#### Prerequisites
The project is a Python‑based codebase (Python notebooks + scripts). You’ll need:

- Python 3.10.8+
- Git
- Conda environment manager
- Google Colab for access to GPUs

#### Clone the repo
Go to your terminal and execute the following commands:

`git clone https://github.com/bart-custers/actuarial_agents.git`

`cd actuarial_agents`

#### Create the virtual environment
For executing the workflow on a local machine, a virtual environment can be created that contains all needed packages and extensions. Go to your terminal and execute the following commands:

`conda create -n actuarial_agents`

`conda activate actuarial_agents`

For executing the workflow in Google Colab the virtual environment is not strictly necessary.

## 2. Connecting the LLM backend

#### Prerequisites
Before connecting to a LLM backend, ensure you have:

- A Hugging Face account and access token (for specific models you want to connect).
- Optional: Google Drive storage for caching large models, but this functionality can be ignored in the `llms/wrappers.py` file.

#### Hugging Face account
1. Create a Hugging Face account: https://huggingface.co/join
2. Generate an access token:
   - Go to Settings → Access Tokens → New token
   - Select read scope
   - Copy the token
3. Store your token in a .env file:
   - `HF_TOKEN=your_huggingface_token_here`
4. Specify the correct path to your `.env` file in the `llms/wrapper.py` file.
   - `load_dotenv("/path/to/.env")`

The `llms/wrapper.py` file will then load the HF_TOKEN from your secret `.env` file for authentication.

#### Setting up Model Cache (optional but recommended)
To connect your Google Drive, specify the path in the `llms/wrapper.py` file:

`model_cache_dir = "/path/to/model_cache"`

The wrapper currently supports the following backends:

- llama7b → `meta-llama/Llama-2-7b-chat-hf`
- llama31_8b → `Meta-Llama-3.1-8B-Instruct`
- qwen25_7b → `Qwen/Qwen2.5-7B-Instruct`
- mock → for testing without a real LLM

The wrapper automatically downloads the model to the cache directory and initializes a HuggingFacePipeline for text generation.

## 3. Executing the workflow
To execute the workflow, you can use the notebook `notebooks/01_workflow_demo.ipynb`.

*Execute on Google Colab:*
1. Run part 1 (Setup) to connect to the Git repo and install it. The code also installs the necessary packages.
2. Run part 2 (Workflow Demo) to execute the workflow. The first part will instantiate the Central Hub and load the chosen LLM backend. Connect to Google Drive when prompted.

*Execute on a local machine:*
1. Run part 2 (Workflow Demo) to execute the workflow. The first part will instantiate the Central Hub and load the chosen LLM backend. Connect to Google Drive when prompted.