import os
import json
import logging
import pandas as pd
import gradio as gr
import multiprocessing

from src.backend import pull_search_results
from src.envs import (
    API, REPO_ID, START_COMMIT_ID,
    HF_CACHE_DIR, SUBMIT_INFOS_SAVE_PATH,
    HF_SEARCH_RESULTS_REPO_DIR, HF_EVAL_RESULTS_REPO_DIR,
    UNZIP_TARGET_DIR,
    TIME_DURATION,
    EVAL_K_VALUES,
    SUBMIT_INFOS_TABLE_COLS
)
from src.css_html_js import custom_css

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# def restart_space():
#     API.restart_space(repo_id=REPO_ID)


def load_submit_infos_df():
    if os.path.exists(SUBMIT_INFOS_SAVE_PATH):
        with open(SUBMIT_INFOS_SAVE_PATH, 'r', encoding='utf-8') as f:
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
            EVAL_K_VALUES,
            HF_CACHE_DIR,
            TIME_DURATION,
            START_COMMIT_ID,
        ),
    )
    process.start()
    demo.launch()
