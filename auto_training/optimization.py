# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import warnings
import optuna
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from auto_training.config_factories.pvt_config_factory import make_pvt_cfg
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector, single_gpu_test
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root, build_dp)


def train(res, augmentation_index, repeat_times):
    timestamp = f"det_res{res}_aug{augmentation_index}_rep{repeat_times}"
    dataset = 'general_hp_opt'
    # dataset = 'small_dataset'
    data_path = f'/data/{dataset}/dataset/coco_data'
    out_path = f'/data/optimization/{dataset}'
    os.makedirs(out_path, exist_ok=True)
    # replace the ${key} with the value of cfg.key
    cfg = make_pvt_cfg(data_path,
                       input_res=(res, res),
                       max_epochs=12,
                       aug_index=augmentation_index,
                       times=repeat_times)
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True


    cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    distributed = False


    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.work_dir = os.path.join(out_path, timestamp)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # dump config
    cfg.dump(osp.join(cfg.work_dir, f"{timestamp}_config.py"))
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()


    # set random seeds
    seed = init_random_seed(None, device=cfg.device)
    seed = seed + dist.get_rank() if None else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {None}')
    set_random_seed(seed, deterministic=None)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = "pvt"


    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        assert 'val' in [mode for (mode, _) in cfg.workflow]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.get(
            'pipeline', cfg.data.train.dataset.get('pipeline'))
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not None),
        timestamp=timestamp,
        meta=meta)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_dataloader_default_args)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    outputs = single_gpu_test(model, data_loader)
    metric = dataset.evaluate(outputs)
    logger.info(
        f"Epoch(test) [{12}][{12}] "
        f"\t {', '.join(f'{key}: {value}' for key, value in metric.items())}"
    )
    return metric


def objective(trial):
    # Define the hyperparameter search space
    res = trial.suggest_categorical("resolution", [384, 512, 768])  # square resolutions
    augmentation_index = trial.suggest_categorical("augmentation_index", [0, 1, 2, 3, 4])  # Indices for different augmentations
    repeat_times = trial.suggest_categorical("repeat_times", [4, 8])  # Repeat dataset times
    print("Resolution", res)
    print("Augmentation index", augmentation_index)
    print("Repeat times", repeat_times)

    # Call the training function with the sampled hyperparameters
    f1_score = 0

    try:
        metric = train(res, augmentation_index, repeat_times)
        f1_score = metric["f1-score"]
    except Exception as e:
        print(f"Error with params: res - {res}, aug_ind - {augmentation_index}, repeat_times - {repeat_times} \n {e}")
    # Return the metric to be maximized or minimized (e.g., accuracy to maximize)
    return f1_score

def main():
    # Create a study object for optimization
    study = optuna.create_study(direction="maximize")  # Use "minimize" if optimizing for loss

    # Run the optimization
    study.optimize(objective, n_trials=35)  # Number of trials to explore

    # Print the best parameters and the best value
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)


if __name__ == '__main__':
    main()
