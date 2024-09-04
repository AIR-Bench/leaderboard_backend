import os
import time
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

LOG_DIR = os.path.join(CACHE_PATH, "logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, f"backend_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")

API = HfApi(token=HF_TOKEN)

HF_SEARCH_RESULTS_REPO_DIR = os.path.join(CACHE_PATH, "search_results")
HF_EVAL_RESULTS_REPO_DIR = os.path.join(CACHE_PATH, "eval_results")

UNZIP_TARGET_DIR = os.path.join(CACHE_PATH, "unzip_target_dir")

TIME_DURATION = 300  # seconds

EVAL_K_VALUES = [1, 3, 5, 10, 50, 100, 1000]
