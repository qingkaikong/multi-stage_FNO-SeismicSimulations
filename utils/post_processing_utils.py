import numpy as np
import os
import matplotlib.pyplot as plt
import torch

l2_loss = lambda diff, y: np.linalg.norm(diff, 2)/np.linalg.norm(y, 2)

def read_training_hist(path):
    train_hist = np.loadtxt(os.path.join(path, 'training_history.csv'), delimiter=',')
    
    return train_hist[train_hist[:, 0] > 0]

def plot_training_traces(hist, color, label=None, alpha=1, only_val=False, linestyle=None):
    field = 2
    
    # if only plot validation
    if not only_val:
        plt.plot(hist[:, 0], hist[:, field-1], color = f'{color}', alpha=alpha, linestyle=':')
    
    plt.plot(hist[:, 0], hist[:, field], 
             f'{color}', label=f'{label}', alpha=alpha, linestyle=linestyle)
    
def return_to_time(data_in_freq, freqs, freq_to_keep, n_after_padding=51, NF=11):
    '''
    Input size: (N*NF, S, S, S, 6), the last dimension is (Ux, Uy, Uz) (complex)
    Return: (N, S, S, S, T, 3), the last dimension is (Ux, Uy, Uz) (real)
    '''
    data_in_time = data_in_freq.view(-1, NF, data_in_freq.size(-4), data_in_freq.size(-3), data_in_freq.size(-2), data_in_freq.size(-1))  # (N, NF, S, S, S, Vel*Cplx)
    #print(data_in_time.shape)
    data_in_time = data_in_time.view(data_in_time.size(0), data_in_time.size(1), data_in_time.size(2), data_in_time.size(3), data_in_time.size(4), 3, 2)  # (N, NF, S, S, S, Vel, Cplx)
    #print(data_in_time.shape)
    data_in_time = data_in_time.permute(0, 2, 3, 4, 5, 1, 6).contiguous()  # (N, S, S, S, Vel, NF, Cplx)
    #print(data_in_time.shape)
    data_in_time = torch.view_as_complex(data_in_time)  # (N, S, S, S, Vel, NF)
    #print(data_in_time.shape)
    kept_freq = torch.zeros(data_in_time.size(0),
                            data_in_time.size(1),
                            data_in_time.size(2),
                            data_in_time.size(3),
                            data_in_time.size(4),
                            len(freqs), dtype=torch.cfloat)
    #print(kept_freq.shape)
    kept_freq[:, :, :, :, :, freq_to_keep] = data_in_time[:, :, :, :, :, :]
    #print(kept_freq.shape)
    data_in_time = torch.fft.irfft(kept_freq.to('cuda:0'), n=n_after_padding, dim=-1, norm='backward')  #(N, S, S, S, Vel, T)
    data_in_time = data_in_time[:, :, :, :, :, :n_after_padding]
    #print(data_in_time.shape)
    data_in_time = data_in_time.permute(0, 1, 2, 3, 5, 4)
    #print(data_in_time.shape)
    #print('Done')
    return data_in_time

def convert_to_time_out(y, NF, n_after_padding, freqs, freq_to_keep):
    out = []
    n = y.size(0) // NF
    for i in range(n):
        x = y[i*NF:(i+1)*NF]
        output_in_time = return_to_time(x, freqs=freqs, freq_to_keep=freq_to_keep, n_after_padding=n_after_padding)
        out.append(output_in_time.detach().cpu().numpy())
    
    return np.vstack(out)
    
def plot_comparison_with_depth_single_freq(results, y_norm, component=0):
    fig = plt.figure(figsize=(12, 6))
    for i, d in enumerate([10, 20, 30, 40, 50]):
        plt.subplot(3, 5, i+1)
        v = np.max(np.abs(y_norm[ix, :, :, d, component]))
        sc=plt.imshow(results[ix, :, :, d, component].T, cmap=plt.cm.seismic, vmin=-v, vmax=v, interpolation=interp_method)
        plt.colorbar(sc)
        plt.title(f"depth={d}")
        if i == 0:
            plt.ylabel('Estimation')

        plt.subplot(3, 5, i+5+1)
        v = np.max(np.abs(y_norm[ix, :, :, d, component]))
        sc=plt.imshow(y_norm[ix, :, :, d, component].T, cmap=plt.cm.seismic, vmin=-v, vmax=v, interpolation=interp_method)
        plt.colorbar(sc)
        plt.title(f"depth={d}")
        if i == 0:
            plt.ylabel('True')

        plt.subplot(3, 5, i+10+1)
        if i == 0:
            plt.ylabel('Residual')
        diff = y_norm[ix, :, :, d, component]-results[ix, :, :, d, component]

        err = l2_loss(diff, y_norm[ix, :, :, d, component])
        # clip the value at 50% of the max true value
        #v = np.max(y[ix, 0, :, :, depth_ix, t, 0]) *1
        sc=plt.imshow(diff.T, cmap=plt.cm.seismic, vmin=-v, vmax=v, interpolation=interp_method)
        plt.title(f'L2 loss: {err:.3f}')
        plt.colorbar(sc)

    #fig.text(0.5, 0.00, f"Source at x {srcx:.1f}, y {srcy:.1f}, z {srcz:.1f}", ha='center', fontsize=16)
    plt.tight_layout()
    plt.show()    
