import os
from huggingface_hub import HfApi


# Info to change for your repository
# ----------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # A read/write token for your org
START_COMMIT_ID = os.environ.get("START_COMMIT_ID", None)

OWNER = "AIR-Bench"  # "nan"  # Change to your org - don't forget to create a results and request dataset, with the correct format!
# ----------------------------------

REPO_ID = f"{OWNER}/leaderboard_backend"
# repo for storing the evaluation results
RESULTS_REPO = f"{OWNER}/eval_results"
# repo for submitting the evaluation
SEARCH_RESULTS_REPO = f"{OWNER}/search_results"

# If you setup a cache later, just change HF_HOME
CACHE_PATH = os.getenv("HF_HOME", ".")
HF_CACHE_DIR = os.path.join(CACHE_PATH, ".cache")
ZIP_CACHE_DIR = os.path.join(CACHE_PATH, ".zip_cache")

SUBMIT_INFOS_SAVE_PATH = os.path.join(CACHE_PATH, "submit_infos.json")

API = HfApi(token=HF_TOKEN)

HF_SEARCH_RESULTS_REPO_DIR = os.path.join(CACHE_PATH, "search_results")
HF_EVAL_RESULTS_REPO_DIR = os.path.join(CACHE_PATH, "eval_results")

UNZIP_TARGET_DIR = os.path.join(CACHE_PATH, "unzip_target_dir")

TIME_DURATION = 300  # seconds

EVAL_K_VALUES = [1, 3, 5, 10, 50, 100, 1000]

def model_hyperlink(link, model_name):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'


def make_clickable_model(model_name: str, model_link: str):
    # link = f"https://huggingface.co/{model_name}"
    if not model_link or not model_link.startswith("https://"):
        return model_name
    return model_hyperlink(model_link, model_name)

SUBMIT_INFOS_TABLE_COLS = ['#', 'Status', 'Retrieval Method', 'Reranking Method', 'Submission Date', 'Revision']
