import os
import torch.nn.functional as F
import numpy as np
import torch
import odl
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def limited_view_parallel_beam_geometry(space, beam_num_angle):
    corners = space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))
    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side
    num_px_horiz = 2 * int(np.ceil(rho * omega / np.pi)) + 1
    det_min_pt = -rho
    det_max_pt = rho
    det_shape = num_px_horiz
    angle_partition = odl.uniform_partition(0, beam_num_angle*(np.pi/180), beam_num_angle)
    det_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)
    return odl.tomo.Parallel2dGeometry(angle_partition, det_partition)

'''
    blocks reconstructions
'''

def next_step_update(dataset, data_tensor, block, device, flag='train'):
    X_ = []
    with torch.no_grad():
        dataloader = DataLoader(deepcopy(data_tensor), batch_size=64, shuffle=False, drop_last=False)
        for (data, grad, target) in dataloader:
            data, grad, target = data.to(device), grad.to(device), target.to(device)
            block.eval()
            X_.append(block.forward(data, grad)[0])
        dataset.update(torch.cat(X_).cpu(), flag=flag)

        info = {'RMSE': torch.sqrt(F.mse_loss(dataset.X_[flag], dataset.data[flag][1])).cpu().numpy()}
        return info

def get_stats(dataset, blocks_history, device, dir_path, save_data=True):
    start = time.time()
    with torch.no_grad():
        mc_samples_mean, mc_samples_var = [], []
        for _ in range(100):
            for block in blocks_history['block']:
                block.eval()
                test_tensor = dataset.construct(flag='test', display=False)
                dataloader = DataLoader(deepcopy(test_tensor), batch_size=64, shuffle=False, drop_last=False)
                X_, Var_ = [], []
                for batch_idx, (data, grad, target) in enumerate(dataloader):
                    data, grad, target = data.to(device), grad.to(device), target.to(device)
                    output, var = block.forward(data, grad)
                    X_.append(output); Var_.append(var)
                dataset.update(torch.cat(X_).cpu(), flag='test')
            mc_samples_mean.append(dataset.X_['test'])
            mc_samples_var.append(torch.cat(Var_).cpu())
            dataset.reset('test')

        print('time: {}'.format(time.time() - start))
        mean = torch.mean(torch.stack(mc_samples_mean), dim=0)

        if hasattr(block, 'bayes_CNN_log_std'):
            std = torch.sqrt( torch.std(torch.stack(mc_samples_mean), dim=0)**2 + torch.mean(torch.stack(mc_samples_var), dim=0) )
        else:
            std = torch.sqrt( torch.std(torch.stack(mc_samples_mean), dim=0)**2 + torch.mean(torch.stack(mc_samples_var), dim=0)[-1] )

        if save_data:
            filename = 'data' + '.p'
            filepath = os.path.join(dir_path, filename)
            with open(filepath, 'wb') as handle:
                pickle.dump({'mean': mean, 'std': std}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        filename = 'numerical_results' + '.txt'
        filepath = os.path.join(dir_path, filename)
        f = open(filepath, 'a')

        psnr_ = []
        for idx, (m, s) in enumerate(zip(mean, std)):
            # simple_visualise(m, os.path.join(dir_path, 'mean_' + str(idx) + '.png'), gray=True)
            # simple_visualise(s, os.path.join(dir_path, 'std_' + str(idx) + '.png'), gray=False)
            psnr = compute_psnr(dataset.data['test'][1][idx].squeeze(dim=0).numpy(), m.squeeze(dim=0).numpy(), data_range=1)
            psnr_.append(psnr)
            string_out = 'img idx {} - PSNR: {:.4f}'.format(idx, psnr)
            f.write(string_out + '\n')
            print('img idx {} - PSNR: {:.4f}'.format(idx, psnr), flush=True)

        string_out = 'average PSNR: {:.4f}'.format( np.mean(psnr_) )
        f.write(string_out + '\n\n')
        print(string_out, flush=True)

def save_net(block, filepath):
     torch.save(block.state_dict(), filepath)

'''
    visualizer - logger
'''

from matplotlib.colors import LinearSegmentedColormap

cm_data = [
    [0.2081, 0.1663, 0.5292],
    [0.2116238095, 0.1897809524, 0.5776761905],
    [0.212252381, 0.2137714286, 0.6269714286],
    [0.2081, 0.2386, 0.6770857143],
    [0.1959047619, 0.2644571429, 0.7279],
    [0.1707285714, 0.2919380952, 0.779247619],
    [0.1252714286, 0.3242428571, 0.8302714286],
    [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429],
    [0.0059571429, 0.4086142857, 0.8828428571],
    [0.0165142857, 0.4266, 0.8786333333],
    [0.032852381, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429],
    [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467],
    [0.0779428571, 0.5039857143, 0.8383714286],
    [0.079347619, 0.5200238095, 0.8311809524],
    [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429],
    [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.819852381],
    [0.0265, 0.6137, 0.8135],
    [0.0238904762, 0.6286619048, 0.8037619048],
    [0.0230904762, 0.6417857143, 0.7912666667],
    [0.0227714286, 0.6534857143, 0.7767571429],
    [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.743552381],
    [0.0589714286, 0.6837571429, 0.7253857143],
    [0.0843, 0.6928333333, 0.7061666667],
    [0.1132952381, 0.7015, 0.6858571429],
    [0.1452714286, 0.7097571429, 0.6646285714],
    [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048],
    [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143],
    [0.3481666667, 0.7424333333, 0.5472666667],
    [0.3952571429, 0.7459, 0.5244428571],
    [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905],
    [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762],
    [0.609852381, 0.7473142857, 0.4336857143],
    [0.6473, 0.7456, 0.4188],
    [0.6834190476, 0.7434761905, 0.4044333333],
    [0.7184095238, 0.7411333333, 0.3904761905],
    [0.7524857143, 0.7384, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286],
    [0.8185047619, 0.7327333333, 0.3497904762],
    [0.8506571429, 0.7299, 0.3360285714],
    [0.8824333333, 0.7274333333, 0.3217],
    [0.9139333333, 0.7257857143, 0.3062761905],
    [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.266647619],
    [0.9937714286, 0.7454571429, 0.240347619],
    [0.9990428571, 0.7653142857, 0.2164142857],
    [0.9955333333, 0.7860571429, 0.196652381],
    [0.988, 0.8066, 0.1793666667],
    [0.9788571429, 0.8271428571, 0.1633142857],
    [0.9697, 0.8481380952, 0.147452381],
    [0.9625857143, 0.8705142857, 0.1309],
    [0.9588714286, 0.8949, 0.1132428571],
    [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661, 0.9514428571, 0.0755333333],
    [0.9763, 0.9831, 0.0538],
    ]
parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

def simple_visualise(x, filename, gray):
    cmap =  cm.Greys_r if gray else parula_map
    plt.imshow(x.squeeze(dim=0), cmap=cmap)
    plt.savefig(filename)

def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped

class TrainVisualiser:
    def __init__(self, filename):
        self.filename = filename
        self.img_ = {}
        self.idx = -1

    @counted
    def update(self, dataset, flag):
        if self.update.calls==1:
            target = dataset.data[flag][1][self.idx]
            self.img_.update({'original': target.squeeze().cpu().numpy()})
            y = dataset.data[flag][0][self.idx]
            self.img_.update({'data': y.squeeze().cpu().numpy()})

        x = dataset.X_[flag][self.idx]
        x = x.squeeze().cpu().numpy()
        self.img_.update({str(self.update.calls): x})

    def generate(self):
        csfont = {'fontname':'Times New Roman'}
        fig, axes = plt.subplots(nrows=1, ncols=len(self.img_))
        for key, ax in zip(self.img_, axes.flatten()):
            ax.imshow(self.img_[key],  cmap=cm.Greys_r)
            if key in ['original', 'data']:
                ax.set_title(key, **csfont)
            else:
                ax.set_title('iter: ' + key, **csfont)
            ax.axis('off')
        fig.savefig(self.filename + '.png', format='png', dpi=900)
        fig.savefig(self.filename + '.svg', format='svg', dpi=900)
        plt.close()
