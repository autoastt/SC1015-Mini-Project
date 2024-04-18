import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from collections import Counter
from scipy.ndimage import convolve1d

def LDS(df): 
    # find effective label distribution
    bin_index_per_label = [get_bin_idx(label) for label in df['popularity']]
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    lds_kernel_window = get_lds_kernel_window()
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
    # mapping lDS to data
    bins = np.linspace(0, 100, num=11) 
    df['bin_index'] = np.digitize(df['popularity'], bins) - 1
    df['bin_index'] = df['bin_index'].clip(0, 10-1)   
    total_samples = len(df)  
    weights = 1 / eff_label_dist  
    weights_normalized = weights / weights.sum() * total_samples
    df['weight'] = df['bin_index'].map(lambda x: weights_normalized[x])
    df = df.drop(['bin_index'],axis=1)
    return df

def get_bin_idx(label, num_bins=10):
    bin_width = 100 / num_bins
    bin_idx = int(label // bin_width)
    return min(bin_idx, num_bins - 1)

def get_lds_kernel_window(kernel = 'gaussian', ks=5, sigma=2):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window