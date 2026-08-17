"""
Evaluate the fine-tuned ResNet checkpoint on the held-out test split (list_test.txt,
36 sequences never seen during fine-tuning) and save per-sequence trajectory plots.
"""
import argparse
import os.path as osp

import main
from main import test_sequence

REPO_ROOT = osp.dirname(osp.abspath(__file__))  # .../IMUNet/RONIN_torch
IMUNET_ROOT = osp.dirname(REPO_ROOT)  # .../IMUNet
DATASET_DIR = osp.join(IMUNET_ROOT, 'Datasets/proposed/IMUNet_dataset')
MODEL_PATH = osp.join(REPO_ROOT, 'Train_out/ResNet/imunet_dataset_ft/checkpoints/checkpoint_best.pt')
# No init_rotor here on purpose: this simulates real-time inference on the phone, where there is
# no ARCore groundtruth orientation available. The earlier run with init_rotor on (kept under
# imunet_dataset_ft/) is the "oracle" baseline for comparison, not the deployment scenario.
OUT_DIR = osp.join(REPO_ROOT, 'Test_out/imunet_dataset_ft_no_init_rotor')
CACHE_DIR = osp.join(REPO_ROOT, 'cache/imunet_dataset_ft_rv_no_initrotor')

args = argparse.Namespace(
    train_list='',
    val_list='',
    test_list=osp.join(DATASET_DIR, 'list_test.txt'),
    test_path=None,
    root_dir=DATASET_DIR,
    cache_path=CACHE_DIR,
    dataset='proposed',
    max_ori_error=20.0,
    ori_source='rv',
    no_init_rotor=True,
    step_size=10,
    window_size=200,
    mode='test',
    lr=2e-5,
    batch_size=128,
    epochs=0,
    arch='ResNet',
    cpu=True,
    run_ekf=False,
    fast_test=False,
    show_plot=False,
    test_status='seen',
    continue_from=None,
    out_dir=OUT_DIR,
    model_path=MODEL_PATH,
    feature_sigma=0.00001,
    target_sigma=0.00001,
)

main.args = args
test_sequence(args)
