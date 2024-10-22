import os
import json
import time
import shutil
import logging
import zipfile
from typing import List, Optional
from collections import defaultdict

from air_benchmark.tasks.tasks import check_benchmark_version
from air_benchmark.evaluation_utils.data_loader import DataLoader
from air_benchmark.evaluation_utils.evaluator import Evaluator

from src.envs import (
    API,
    ZIP_CACHE_DIR,
    SEARCH_RESULTS_REPO, RESULTS_REPO, SUBMIT_INFOS_REPO,
    make_clickable_model
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s | %(name)s | %(levelname)s: %(message)s',
    force=True
)


def compute_metrics(
    benchmark_version: str,
    search_results_save_dir: str,
    k_values: List[int] = [1, 3, 5, 10, 50, 100, 1000],
    cache_dir: Optional[str] = None,
):
    data_loader = DataLoader(benchmark_version, cache_dir=cache_dir)
    evaluator = Evaluator(data_loader)
    
    eval_results = evaluator.evaluate_results(search_results_save_dir, k_values=k_values, split='test')
    return eval_results


def save_evaluation_results(
    eval_results: dict, 
    save_path: str,
    model_name: str,
    reranker_name: str,
    model_link: Optional[str] = None,
    reranker_link: Optional[str] = None,
    is_anonymous: bool = False,
    timestamp: str = None,
    revision: str = None,
):
    results = defaultdict(list)
    configs = {}
    
    for task_type, task_type_results in eval_results.items():
        for domain, domain_results in task_type_results.items():
            for lang, lang_results in domain_results.items():
                for dataset_name, task_results in lang_results.items():
                    for metric, metric_val in task_results.items():
                        _key = f"{model_name}_{reranker_name}_{task_type}_{metric}"
                        results[_key].append({
                            "domain": domain,
                            "lang": lang,
                            "dataset": dataset_name,
                            "value": metric_val,
                        })
                        configs[_key] = {
                            "retrieval_model": model_name,
                            "retrieval_model_link": model_link,
                            "reranking_model": reranker_name,
                            "reranking_model_link": reranker_link,
                            "task": task_type,
                            "metric": metric,
                            "timestamp": timestamp,
                            "is_anonymous": is_anonymous,
                            "revision": revision,
                        }
    
    results_list = []
    for k, result in results.items():
        config = configs[k]
        results_list.append({
            "config": config,
            "results": result
        })
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)


def get_file_list(dir_path: str, allowed_suffixes: List[str] = None) -> List[str]:
    file_paths = set()
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        for root, _, files in os.walk(dir_path):
            for file in files:
                if allowed_suffixes is None or any(
                    file.endswith(suffix) for suffix in allowed_suffixes
                ):
                    file_paths.add(os.path.abspath(os.path.join(root, file)))
    return file_paths


def get_zip_file_path(zip_file_name: str):
    zip_file_path = None
    for root, _, files in os.walk(ZIP_CACHE_DIR):
        for file in files:
            if file == zip_file_name:
                zip_file_path = os.path.abspath(os.path.join(root, file))
                break
    return zip_file_path


def find_file(file_name: str, dir_path: str):
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return False
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file == file_name:
                return True
    return False


def get_submit_infos_list(file_paths: List[str], eval_results_dir: str) -> dict:
    submit_infos_list = []
    for file_path in file_paths:
        submit_info = {
            'ID': None,
            'Submission Date': None,
            'Benchmark Version': None,
            'Retrieval Method': None,
            'Reranking Method': None,
            'Revision': None,
            'Status': None,
        }
        file_name = os.path.basename(file_path).split('.')[0]
        rank_time = file_name.split('-')[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        submit_info['ID'] = rank_time
        submit_info['Submission Date'] = metadata['timestamp']
        submit_info['Benchmark Version'] = metadata['version']
        submit_info['Retrieval Method'] = (make_clickable_model(metadata['model_name'], metadata['model_url']))
        submit_info['Reranking Method'] = (make_clickable_model(metadata['reranker_name'], metadata['reranker_url']))
        submit_info['Revision'] = metadata['revision']
        if find_file(f"results_{file_name}.json", eval_results_dir):
            submit_info['Status'] = "✔️ Success"
        else:
            submit_info['Status'] = "❌ Failed"
        submit_infos_list.append(submit_info)
    sorted_submit_infos_list = sorted(submit_infos_list, key=lambda x: x['ID'])
    for order, submit_info in enumerate(sorted_submit_infos_list, 1):
        submit_info['ID'] = order
    rsorted_submit_infos_list = sorted(sorted_submit_infos_list, key=lambda x: x['ID'], reverse=True)
    return rsorted_submit_infos_list


def pull_search_results(
    hf_search_results_repo_dir: str,
    hf_eval_results_repo_dir: str,
    unzip_target_dir: str,
    submit_infos_dir: str,
    submit_infos_file_name: str = "submit_infos.json",
    k_values: List[int] = [1, 3, 5, 10, 50, 100, 1000],
    cache_dir: str = None,
    time_duration: int = 1800,
    start_commit_id: str = None
):
    # Pull the submit infos
    API.snapshot_download(
        repo_id=SUBMIT_INFOS_REPO,
        repo_type="dataset",
        local_dir=submit_infos_dir,
        etag_timeout=30
    )
    
    logger.warning(f"Start from commit: {start_commit_id}")
    if start_commit_id is not None:
        API.snapshot_download(
            repo_id=SEARCH_RESULTS_REPO,
            repo_type="dataset",
            revision=start_commit_id,
            local_dir=hf_search_results_repo_dir,
            etag_timeout=30,
            allow_patterns=['*.json']
        )
        cur_file_paths = get_file_list(hf_search_results_repo_dir, allowed_suffixes=['.json'])
    else:
        API.snapshot_download(
            repo_id=SEARCH_RESULTS_REPO,
            repo_type="dataset",
            local_dir=hf_search_results_repo_dir,
            etag_timeout=30,
            allow_patterns=['*.json']
        )
        cur_file_paths = get_file_list(hf_search_results_repo_dir, allowed_suffixes=['.json'])
    
    logger.warning("Start to pull new search results ...")
    while True:
        os.makedirs(ZIP_CACHE_DIR, exist_ok=True)
        os.makedirs(unzip_target_dir, exist_ok=True)
        try:
            API.snapshot_download(
                repo_id=RESULTS_REPO,
                repo_type="dataset",
                local_dir=hf_eval_results_repo_dir,
                etag_timeout=30
            )
            API.snapshot_download(
                repo_id=SEARCH_RESULTS_REPO,
                repo_type="dataset",
                local_dir=hf_search_results_repo_dir,
                etag_timeout=30,
                allow_patterns=['*.json']
            )
        except Exception as e:
            logger.error(f"Failed to download the search results or evaluation results: {e}")
            logger.error(f"Wait for {time_duration} seconds for the next update ...")
            time.sleep(time_duration)
            continue
        
        commit_infos_dict = defaultdict(list)
        
        new_file_paths = get_file_list(hf_search_results_repo_dir, allowed_suffixes=['.json'])
        added_file_paths = new_file_paths - cur_file_paths
        for metadata_file_path in sorted(list(added_file_paths)):
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            model_name = metadata['model_name']
            model_link = None if not metadata['model_url'] else metadata['model_url']
            reranker_name = metadata['reranker_name']
            reranker_link = None if not metadata['reranker_url'] else metadata['reranker_url']
            benchmark_version = metadata['version']
            
            try:
                check_benchmark_version(benchmark_version)
            except ValueError:
                logger.error(f"Invalid benchmark version `{benchmark_version}` in file `{metadata_file_path}`. Skip this commit.")
                continue
            
            file_name = os.path.basename(metadata_file_path).split('.')[0]
            zip_file_name = f"{file_name}.zip"
            
            try:
                API.snapshot_download(
                    repo_id=SEARCH_RESULTS_REPO,
                    repo_type="dataset",
                    local_dir=ZIP_CACHE_DIR,
                    etag_timeout=30,
                    allow_patterns=[f'*{zip_file_name}']
                )
                zip_file_path = get_zip_file_path(zip_file_name)
                assert zip_file_path is not None, f"zip_file_path is None"
            except Exception as e:
                logger.error(f"Failed to download the zip file `{zip_file_name}`: {e}")
                continue

            unzip_target_path = os.path.join(unzip_target_dir, benchmark_version, file_name)
            os.makedirs(unzip_target_path, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_target_path)
            except Exception as e:
                logger.error(f"Failed to unzip the search results `{file_name}`: {e}")
                continue
            
            commit_infos_dict[benchmark_version].append({
                "model_name": model_name,
                "model_link": model_link,
                "reranker_name": reranker_name,
                "reranker_link": reranker_link,
                "is_anonymous": metadata['is_anonymous'],
                "file_name": file_name,
                "timestamp": metadata['timestamp'],
                "revision": metadata['revision'],
                "search_results_dir": unzip_target_path
            })
        
        # Sort the search results by timestamp
        for benchmark_version in commit_infos_dict:
            commit_infos_dict[benchmark_version].sort(key=lambda x: int(os.path.basename(x["search_results_dir"]).split('-')[0]))
        
        # Save the evaluation results
        update_flag = False
        new_models_set = set()
        for benchmark_version, commit_infos in commit_infos_dict.items():
            eval_results_dir = os.path.join(hf_eval_results_repo_dir, benchmark_version)
            os.makedirs(eval_results_dir, exist_ok=True)
            
            for commit_info in commit_infos:
                try:
                    eval_results = compute_metrics(
                        benchmark_version, 
                        commit_info['search_results_dir'],
                        k_values=k_values,
                        cache_dir=cache_dir,
                    )
                except KeyError as e:
                    logger.error(f"KeyError: {e}. Skip this commit: {commit_info['file_name']}")
                    continue
                
                save_dir = os.path.join(eval_results_dir, commit_info['model_name'], commit_info['reranker_name'])
                os.makedirs(save_dir, exist_ok=True)
                results_save_path = os.path.join(save_dir, f"results_{commit_info['file_name']}.json")
                save_evaluation_results(eval_results, 
                                        results_save_path,
                                        commit_info['model_name'],
                                        commit_info['reranker_name'],
                                        model_link=commit_info['model_link'],
                                        reranker_link=commit_info['reranker_link'],
                                        is_anonymous=commit_info['is_anonymous'],
                                        timestamp=commit_info['timestamp'],
                                        revision=commit_info['revision'])
                new_models_set.add(f"{commit_info['model_name']}_{commit_info['reranker_name']}")
                
                update_flag = True
        
        # Commit the updated evaluation results
        if update_flag:
            commit_message = "Update evaluation results\nNew models added in this update:\n"
            for new_model in new_models_set:
                commit_message += f"\t- {new_model}\n"
            
            API.upload_folder(
                repo_id=RESULTS_REPO,
                folder_path=hf_eval_results_repo_dir,
                path_in_repo=None,
                commit_message=commit_message,
                repo_type="dataset"
            )
            logger.warning("Evaluation results updated and pushed to the remote repository.")

            # Print the new models
            logger.warning("=====================================")
            logger.warning("New models added in this update:")
            for new_model in new_models_set:
                logger.warning("\t" + new_model)
        
        # Clean the cache
        shutil.rmtree(ZIP_CACHE_DIR)
        shutil.rmtree(unzip_target_dir)
        
        # update submit infos
        if new_file_paths != cur_file_paths:
            cur_file_paths = new_file_paths
            submit_infos_list = get_submit_infos_list(cur_file_paths, hf_eval_results_repo_dir)
            submit_infos_save_path = os.path.join(submit_infos_dir, submit_infos_file_name)
            with open(submit_infos_save_path, 'w', encoding='utf-8') as f:
                json.dump(submit_infos_list, f, ensure_ascii=False, indent=4)
            
            # Commit the updated submit infos
            API.upload_folder(
                repo_id=SUBMIT_INFOS_REPO,
                folder_path=submit_infos_dir,
                path_in_repo=None,
                commit_message="Update submission infos",
                repo_type="dataset"
            )
        
        # Wait for the next update
        logger.warning(f"Wait for {time_duration} seconds for the next update ...")
        
        time.sleep(time_duration)
