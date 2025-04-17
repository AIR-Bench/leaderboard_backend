import os
import json
import logging
import pandas as pd
import gradio as gr
import multiprocessing

from src.backend import pull_search_results
from src.envs import (
    API, START_COMMIT_ID, REPO_ID,
    HF_CACHE_DIR, SUBMIT_INFOS_DIR, SUBMIT_INFOS_FILE_NAME,
    HF_SEARCH_RESULTS_REPO_DIR, HF_EVAL_RESULTS_REPO_DIR, SUBMIT_INFOS_REPO,
    UNZIP_TARGET_DIR,
    TIME_DURATION,
    EVAL_K_VALUES,
    SUBMIT_INFOS_TABLE_COLS,
    TIMEOUT,
)
from src.css_html_js import custom_css

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)


def restart_space():
    API.restart_space(repo_id=REPO_ID)


def load_submit_infos_df():
    # Pull the submit infos
    API.snapshot_download(
        repo_id=SUBMIT_INFOS_REPO,
        repo_type="dataset",
        local_dir=SUBMIT_INFOS_DIR,
        etag_timeout=TIMEOUT,
    )
    submit_infos_save_path = os.path.join(SUBMIT_INFOS_DIR, SUBMIT_INFOS_FILE_NAME)
    
    if os.path.exists(submit_infos_save_path):
        with open(submit_infos_save_path, 'r', encoding='utf-8') as f:
            submit_infos = json.load(f)
    else:
        submit_infos = []
    if submit_infos:
        submit_infos_df = pd.DataFrame(submit_infos)[SUBMIT_INFOS_TABLE_COLS]
    else:
        submit_infos_df = pd.DataFrame(columns=SUBMIT_INFOS_TABLE_COLS)
    return submit_infos_df


with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## Submission Infos Table")
        
    table = gr.components.Dataframe(
        value=load_submit_infos_df(),
        elem_id="submission-infos-table",
        interactive=False,
        datatype="markdown"
    )
        
    refresh_button = gr.Button("Refresh Submission Infos")

    refresh_button.click(
        fn=load_submit_infos_df,
        outputs=table,
    )


if __name__ == "__main__":
    process = multiprocessing.Process(
        target=pull_search_results,
        args=(
            HF_SEARCH_RESULTS_REPO_DIR,
            HF_EVAL_RESULTS_REPO_DIR,
            UNZIP_TARGET_DIR,
            SUBMIT_INFOS_DIR,
            SUBMIT_INFOS_FILE_NAME,
            EVAL_K_VALUES,
            HF_CACHE_DIR,
            TIME_DURATION,
            START_COMMIT_ID,
        ),
    )
    process.start()

    demo.queue(default_concurrency_limit=40)
    demo.launch()
