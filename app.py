import os
import gradio as gr
import multiprocessing

from src.backend import pull_search_results
from src.envs import (
    API, REPO_ID, START_COMMIT_ID,
    LOG_DIR, HF_CACHE_DIR,
    HF_SEARCH_RESULTS_REPO_DIR, HF_EVAL_RESULTS_REPO_DIR,
    UNZIP_TARGET_DIR,
    TIME_DURATION,
    EVAL_K_VALUES,
)

def restart_space():
    API.restart_space(repo_id=REPO_ID)


def get_log_files():
    if not os.path.exists(LOG_DIR):
        return []
    return sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.log')])


def refresh_log_files():
    return get_log_files()


def display_log_content(selected_file):
    if selected_file:
        with open(os.path.join(LOG_DIR, selected_file), 'r', encoding='utf-8') as file:
            return file.read()
    return "No log file selected"


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    
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
    
    with gr.Blocks() as demo:
        gr.Markdown("## Select a log file to view its content")
        
        log_file_dropdown = gr.Dropdown(
            choices=refresh_log_files(),
            label="Select log file",
            interactive=True,
        )
        log_content_box = gr.Textbox(
            label="Log content",
            lines=20,
            interactive=False,
        )
        refresh_button = gr.Button("Refresh log files")
        
        log_file_dropdown.change(
            fn=display_log_content,
            inputs=log_file_dropdown,
            outputs=log_content_box,
        )
        refresh_button.click(
            fn=refresh_log_files,
            outputs=log_file_dropdown,
        )
    
    demo.launch()
