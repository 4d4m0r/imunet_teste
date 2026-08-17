"""
Export the fine-tuned ResNet1D checkpoint directly to ExecuTorch (.pte), for on-device
inference on Android. No ONNX/TFLite involved.

Input:  (1, 6, 200)  float32  -- [gyro_x,y,z, acce_x,y,z] rotated into the device's current
                                  orientation frame (ori = rv(t), no init_rotor), window of
                                  200 samples at 200Hz.
Output: (1, 2)        float32  -- (vx, vy) planar velocity, in the same frame as the input.
"""
import os.path as osp

import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule

REPO_ROOT = osp.dirname(osp.abspath(__file__))
CHECKPOINT = osp.join(REPO_ROOT, 'Train_out/ResNet/imunet_dataset_ft/checkpoints/checkpoint_best.pt')
OUT_PATH = osp.join(REPO_ROOT, 'Test_out/imunet_dataset_ft/model.pte')

WINDOW_SIZE = 200
_fc_config = {'fc_dim': 512, 'in_dim': WINDOW_SIZE // 32 + 1, 'dropout': 0.5, 'trans_planes': 128}

model = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2],
                  base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
checkpoint = torch.load(CHECKPOINT, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print('Loaded checkpoint from epoch', checkpoint.get('epoch'))

example_input = (torch.randn(1, 6, WINDOW_SIZE),)

with torch.no_grad():
    reference_output = model(*example_input)

exported_program = export(model, example_input)
edge_program = to_edge_transform_and_lower(exported_program, partitioner=[XnnpackPartitioner()])
executorch_program = edge_program.to_executorch()

with open(OUT_PATH, 'wb') as f:
    f.write(executorch_program.buffer)

print('Saved', OUT_PATH, '(%d bytes)' % osp.getsize(OUT_PATH))

# Sanity check: run the exported program through ExecuTorch's own runtime and compare
# to the eager PyTorch output for the same input.
from executorch.runtime import Runtime

runtime = Runtime.get()
program = runtime.load_program(OUT_PATH)
method = program.load_method('forward')
outputs = method.execute(example_input)

import numpy as np
pte_out = np.array(outputs[0])
ref_out = reference_output.numpy()
max_abs_diff = np.max(np.abs(pte_out - ref_out))
print('Reference (eager) output:', ref_out)
print('ExecuTorch (.pte) output:', pte_out)
print('Max abs diff:', max_abs_diff)
assert max_abs_diff < 1e-3, 'ExecuTorch export diverges from the eager model output!'
print('OK: .pte output matches eager PyTorch output within tolerance.')
