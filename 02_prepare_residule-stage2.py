from models.fno_3d import FNO3d

from utils.data_utils import load_data_distribute

import numpy as np
import torch
import os
import glob
from pathlib import Path

data_path = './demo_data'
target_freq = 0

print('Loading data')

all_files = sorted(glob.glob(os.path.join(data_path, '*.npz')))
print(f'There are {len(all_files)} files.')

device = torch.device('cuda')

model_dir = f'./output_stage1/'

in_channels = 8
width = 48 
mode = 30

model = FNO3d(mode, mode, mode, width, in_channels=in_channels, debug=False).to(device)

checkpoint = torch.load(f'{model_dir}best_model.pt')
model.load_state_dict(checkpoint['state_dict'])

batch_size = 1
n_workers = 1
distributed = False
test_loader = load_data_distribute(files=all_files, 
                         batch_size=batch_size, 
                         shuffle=False, 
                         n_workers=n_workers, 
                         divide="Test")

residual=[]
output_folder = f'./residual/stage2/residual_freq{target_freq}/'
Path(output_folder).mkdir(parents=True, exist_ok=True)

for i, (x, y) in enumerate(test_loader):
    filename = os.path.basename(all_files[i])
    output_name = f'{output_folder}{filename}'

    if i%1000 == 0:
        print(f"Freq: {target_freq}, {i}")
    
    x, y = x.to(device), y.to(device)
    out = model(x)
    res = out - y
    inp = np.concatenate([x.cpu().detach().numpy()[0], out.cpu().detach().numpy()[0]], axis=-1)
    np.savez_compressed(output_name, x=inp, y=res.cpu().detach().numpy()[0])
