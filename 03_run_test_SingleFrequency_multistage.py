import torch
import numpy as np

from timeit import default_timer
import pickle
torch.manual_seed(0)
np.random.seed(0)

import os
import glob

from utils.data_utils import load_data_distribute
from models.fno_3d import FNO3d

# Note, in this script, I was calling stage 2 model stage 1, because I 
# initially used stage 0 and stage 1, but when write the paper, I changed
# it to stage 1 and stage 2. 

################################################################
# configs
################################################################
batch_size = 1
n_workers = 8

device1 = torch.device('cuda:1')
device2 = torch.device('cuda:2')
################################################################
# load data
################################################################
t1 = default_timer()

# for demo purposes, we use the same demo data for testing
target_freq = 0
data_path = './demo_data'
stage1_test_files = sorted(glob.glob(os.path.join(data_path, '*.npz')))

data_path = f'./residual/stage2/residual_freq{target_freq}/'
stage2_test_files = sorted(glob.glob(os.path.join(data_path, '*.npz')))

stage1_test_loader = load_data_distribute(files=stage1_test_files, 
                         batch_size=batch_size, 
                         shuffle=False, 
                         n_workers=n_workers, 
                         divide="Test")

stage2_test_loader = load_data_distribute(files=stage2_test_files, 
                         batch_size=batch_size, 
                         shuffle=False, 
                         n_workers=n_workers, 
                         divide="Test")


t2 = default_timer()

print('preprocessing finished, time used:', t2 - t1)

################################################################
# training
################################################################
in_channels = 8
width = 48
mode = 30
stage1_model = FNO3d(mode, mode, mode, width, in_channels=8, debug=False).to(device1)

stage2_model = FNO3d(mode, mode, mode, width, in_channels=14, debug=False).to(device2)
stage2_model.eval()
stage1_model.eval()


stage1_model_dir = f'./output_stage1'
stage2_model_dir = f'./output_stage2'

checkpoint = torch.load(f'{stage1_model_dir}/best_model.pt')
stage1_model.load_state_dict(checkpoint['state_dict'])

checkpoint = torch.load(f'{stage2_model_dir}/best_model.pt')
stage2_model.load_state_dict(checkpoint['state_dict'])


# only need one time
y_output_file = f'./test_y.npz'

stage2_results=[]
stage1_results = []
combined_results = []

i = 0
test_y = []
for (x, y), (x_res, y_res) in zip(stage1_test_loader, stage2_test_loader):
    i += 1
    x, y = x.to(device1), y.to(device1)
    x_res, y_res = x_res.to(device2), y_res.to(device2)
    
    test_y.append(y.cpu().detach().numpy())
        
    out_stage1 = stage1_model(x).cpu().detach().numpy()
    out_stage2 = stage2_model(x_res).cpu().detach().numpy()
    
    
    stage1_results.append(out_stage1)
    stage2_results.append(out_stage2)
    combined_results.append(out_stage1-out_stage2)
    
    if i%100 == 0:
        print(i)

stage1_results = np.array(stage1_results)
np.savez_compressed(f'{stage1_model_dir}/stage1_test_results.npz', results=stage1_results)

stage2_results = np.array(stage2_results)
np.savez_compressed(f'{stage2_model_dir}/stage2_test_results.npz', results=stage2_results)

combined_results = np.array(combined_results)
np.savez_compressed(f'{stage2_model_dir}/combined_test_results.npz', results=combined_results)

#if not os.path.exists(y_output_file):
np.savez_compressed(y_output_file, y=test_y)
