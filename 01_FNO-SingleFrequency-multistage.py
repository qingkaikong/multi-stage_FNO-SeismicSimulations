import os

from pathlib import Path
import glob

import numpy as np
 
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from models.fno_3d import FNO3d
from utils.training_utils import *
from utils.distributed_utils import *
from utils.data_utils import *

from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

del os.environ['OMP_PLACES']
del os.environ['OMP_PROC_BIND']

################################################################
# setup Tioga
################################################################
local_rank, rank, world_size = setup_Tioga()

################################################################
# set stage
################################################################
stage = 2

################################################################
# load data
################################################################

batch_size = 4
n_workers = 7

t1 = default_timer()

# since we train single frequency, I will use 0 - 1 to indicate 
# the frequency from 2 - 6 Hz, 0 corresponds to 2 Hz, which I 
# put a few demo data there
target_freq = 0

print('Loading data')

if stage == 1:
    data_path = './demo_data'
elif stage == 2:
    data_path = f'./residual/stage2/residual_freq{target_freq}/'
    
all_files = sorted(glob.glob(os.path.join(data_path, '*.npz')))
print(f'There are {len(all_files)} files.')

n_training = 3

train_files =all_files[:n_training]

train_loader = load_data_distribute(files=train_files, 
                         batch_size=batch_size, 
                         shuffle=True, 
                         n_workers=n_workers, 
                         divide="Train")

val_files = all_files[n_training:]

val_loader = load_data_distribute(files=val_files, 
                         batch_size=batch_size, 
                         shuffle=False, 
                         n_workers=n_workers, 
                         divide="Validation")

t2 = default_timer()

ntrain = len(train_loader.dataset)
nval = len(val_loader.dataset)
print('preprocessing finished, time used:', t2 - t1)
print(f"It loads {ntrain} training data and {nval} validation data")

################################################################
# Training
################################################################
epochs = 5

if stage == 1:
    in_channels = 8
    width = 48
    mode = 30
    learning_rate = 1e-3
    checkpoint_dir = f'./output_stage1/'
    
elif stage == 2:
    in_channels = 14
    width = 48
    mode = 30
    learning_rate = 1e-2
    checkpoint_dir = f'./output_stage2/'
    
loss_str = 'L1L2'

print(f"Epoch {epochs}, width {width}, learning rate {learning_rate}, loss {loss_str}, batchsize {batch_size}")
print(f'Training stage {stage} model')
print(f'Saving model to folder {checkpoint_dir}')

Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
device = torch.device('cuda')

model = FNO3d(mode, mode, mode, width, in_channels=in_channels, debug=False).to(device)
model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
model_without_ddp = model.module
    
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-5)

use_scheduler = True
mix_precision = False

if use_scheduler:
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                   step_size = 30, # Period of learning rate decay
                   gamma = 0.5)
else:
    lr_scheduler = None

train_valDistributed(epochs, model, model_without_ddp, device,
      train_loader, val_loader, 
      optimizer, lr_scheduler,
      checkpoint_dir, 
      epoch_start_to_save_model=1, 
      loss_str=loss_str, 
      mix_precision=mix_precision)

cleanup()
print(rank, "Done", flush=True)