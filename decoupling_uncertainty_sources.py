import sys
sys.path.append('./mean_field_VI')

import os
import argparse
import time
import numpy as np
import pickle
import json
from torch.utils.data import DataLoader
from distutils.util import strtobool
from copy import deepcopy

# torch imports
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# forward block & dataset
import odl
from odl.contrib import torch as odl_torch
from dataset_constructor import DatasetConstructor
from forward_model import SimpleCT, ForwardModel
from dataset import DataSet

# metrics
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

# utils
from utils import TrainVisualiser, next_step_update, get_stats, save_net, limited_view_parallel_beam_geometry
import mean_field_VI.utils as block_utils

def main():

    parser = argparse.ArgumentParser()
    # general & dataset & training settings
    parser.add_argument('--k_max', type=int, default=5,
                        help='Max reconstruction iterations')
    parser.add_argument('--save_figs', type = lambda x:bool(strtobool(x)), default=True,
                        help='save pics in reconstruction')
    parser.add_argument('--img_mode', type=str, default='SimpleCT',
                        help=' image-modality reconstruction: SimpleCT')
    parser.add_argument('--train_size', type=int, default=4000,
                        help='dataset size')
    parser.add_argument('--dataset_type', type=str, default='GenEllipsesSamples',
                        help='GenEllipsesSamples or GenFoamSamples')
    parser.add_argument('--pseudo_inverse_init', type = lambda x:bool(strtobool(x)), default=True,
                        help='initialise with pseudoinverse')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--initial_lr', type=float, default=1e-3,
                        help='initial_lr')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for valing')
    parser.add_argument('--arch_args', type=json.loads, default=dict(),
                        help='load architecture dictionary')
    parser.add_argument('--block_type', type=str, default='bayesian_homo',
                        help='deterministic, bayesian_homo, bayesian_hetero')

    # forward models setting
    parser.add_argument('--size', type=int, default=128,
                        help='image size')
    parser.add_argument('--beam_num_angle', type=int, default=30,
                        help='number of angles / projections')
    parser.add_argument('--limited_view', type = lambda x:bool(strtobool(x)), default=False,
                        help='limited view geometry instead of sparse view geometry')
    # options
    parser.add_argument('--no_cuda', type = lambda x:bool(strtobool(x)), default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=222,
                        help='random seed')
    parser.add_argument('--config', default='configs/bayesian_arch_config.json',
                        help='config file path')

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config) as handle:
            config = json.load(handle)
        vars(args).update(config)

    block_utils.set_gpu_mode(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.img_mode == SimpleCT.__name__:
        img_mode = SimpleCT()
        half_size = args.size / 2
        space =  odl.uniform_discr([-half_size, -half_size],
                                   [half_size, half_size],
                                   [args.size, args.size], dtype='float32')
        img_mode.space = space
        if not args.limited_view:
            geometry = odl.tomo.parallel_beam_geometry(space, num_angles=args.beam_num_angle)
        elif args.limited_view:
            geometry = limited_view_parallel_beam_geometry(space, beam_num_angle=args.beam_num_angle)
        else:
            raise NotImplementedError
        img_mode.geometry = geometry
        operator = odl.tomo.RayTransform(space, geometry)
        opnorm = odl.power_method_opnorm(operator)
        img_mode.operator = odl_torch.OperatorModule((1 / opnorm) * operator)
        img_mode.adjoint = odl_torch.OperatorModule((1 / opnorm) * operator.adjoint)
        pseudoinverse = odl.tomo.fbp_op(operator)
        pseudoinverse = odl_torch.OperatorModule(pseudoinverse * opnorm)
        img_mode.pseudoinverse = pseudoinverse

        geometry_specs = 'full_view_sparse_' + str(args.beam_num_angle) if not args.limited_view else 'limited_view_' + str(args.beam_num_angle)
        dataset_name = 'dataset' + '_' + args.img_mode + '_' + str(args.size) \
        + '_' + str(args.train_size) + '_' + geometry_specs + '_' + args.dataset_type

        data_constructor = DatasetConstructor(img_mode, train_size=args.train_size, dataset_name=dataset_name)
        data = data_constructor.data()
    else:
        raise NotImplementedError
    dataset = DataSet(data, img_mode, args.pseudo_inverse_init)

    optim_parms = {'epochs':args.epochs, 'initial_lr':  args.initial_lr, 'batch_size': args.batch_size}

    if args.block_type == 'deterministic':
        from blocks import DeterministicBlock as Block
    elif args.block_type == 'bayesian_homo':
        from blocks import BlockHomo as Block
    elif args.block_type == 'bayesian_hetero':
        from blocks import BlockHetero as Block
    else:
        raise NotImplementedError

    # results directory
    path = os.path.dirname(__file__)
    dir_path = os.path.join(path, 'results', args.img_mode, args.block_type, args.dataset_type, str(args.train_size), geometry_specs, str(args.size), str(args.seed))
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # all config
    print('===========================\n', flush=True)
    for key, val in vars(args).items():
        print('{}: {}'.format(key, val), flush=True)
    print('===========================\n', flush=True)

    blocks_history = {'block': []}
    start_time = time.time()
    # looping through architecture-blocs
    for idx in range(1, args.k_max + 1):

        print('============== training block number: {} ============= \n'.format(idx), flush=True)

        block = Block(args.arch_args)
        block = block.to(device)

        path_block = os.path.join(dir_path, str(idx) + '.pt')
        if os.path.exists(path_block):
            block.load_state_dict( torch.load(path_block) )
        else:
            raise NotImplementedError

        blocks_history['block'].append(block)

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
            epistemic = torch.std(torch.stack(mc_samples_mean), dim=0)**2
            aleatoric = torch.mean(torch.stack(mc_samples_var), dim=0)
            std = torch.sqrt( torch.std(torch.stack(mc_samples_mean), dim=0)**2 + torch.mean(torch.stack(mc_samples_var), dim=0) )
        else:
            raise NotImplementedError

        dir_path = os.path.join(dir_path, 'uncertainty analysis')
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        filename = 'data' + '_' + str(args.k_max) + '.p'
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'wb') as handle:
            pickle.dump({'mean': mean, 'aleatoric': aleatoric, 'epistemic': epistemic, 'std': std}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    # reconstruction
    resonstruction_dir_path = os.path.join(dir_path, str(idx))
    if not os.path.isdir(resonstruction_dir_path):
        os.makedirs(resonstruction_dir_path)
    get_stats(dataset, blocks_history, device, resonstruction_dir_path)

    print('--- training time: %s seconds ---' % (time.time() - start_time), flush=True)


if __name__ == '__main__':
    main()
