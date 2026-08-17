"""
Fine-tune the pretrained RONIN ResNet checkpoint on the IMUNet_dataset (proposed format),
using init_rotor (default, matches how the checkpoint was trained) and ori_source='rv'
(this dataset has no game_rv columns).

Bypasses main.py's __main__ block, which hardcodes root_dir/train_list/val_list/out_dir
per --dataset choice regardless of CLI overrides -- so this script builds args directly
and calls train()/test_sequence() as a library instead.
"""
import argparse
import os.path as osp

import main
from main import train, test_sequence

REPO_ROOT = osp.dirname(osp.abspath(__file__))  # .../IMUNet/RONIN_torch
IMUNET_ROOT = osp.dirname(REPO_ROOT)  # .../IMUNet
DATASET_DIR = osp.join(IMUNET_ROOT, 'Datasets/proposed/IMUNet_dataset')
CHECKPOINT = osp.join(IMUNET_ROOT, 'Datasets/ronin/ronin_resnet_model/checkpoint_gsn_latest.pt')
OUT_DIR = osp.join(REPO_ROOT, 'Train_out/ResNet/imunet_dataset_ft')
CACHE_DIR = osp.join(REPO_ROOT, 'cache/imunet_dataset_ft_rv_initrotor')

ADDITIONAL_EPOCHS = 30

args = argparse.Namespace(
    train_list=osp.join(DATASET_DIR, 'list_train_ft.txt'),
    val_list=osp.join(DATASET_DIR, 'list_val_ft.txt'),
    test_list=osp.join(DATASET_DIR, 'list_test.txt'),
    test_path=None,
    root_dir=DATASET_DIR,
    cache_path=CACHE_DIR,
    dataset='proposed',
    max_ori_error=20.0,
    ori_source='rv',
    no_init_rotor=False,
    step_size=10,
    window_size=200,
    mode='train',
    lr=2e-5,
    batch_size=128,
    epochs=ADDITIONAL_EPOCHS,  # placeholder, corrected below once start_epoch is known
    arch='ResNet',
    cpu=False,
    run_ekf=False,
    fast_test=False,
    show_plot=False,
    test_status='seen',
    continue_from=CHECKPOINT,
    out_dir=OUT_DIR,
    model_path='',
    feature_sigma=0.00001,
    target_sigma=0.00001,
)

import torch
ckpt = torch.load(args.continue_from, map_location='cpu')
start_epoch = ckpt.get('epoch', 0)
args.epochs = start_epoch + ADDITIONAL_EPOCHS
print('Checkpoint was at epoch {}, training up to epoch {} ({} additional epochs)'.format(
    start_epoch, args.epochs, ADDITIONAL_EPOCHS))

main.args = args
train(args)
