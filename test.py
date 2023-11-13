import argparse
import gc
import os

import numpy as np
import scipy.io as io
import torch

from tqdm import tqdm

from network import PCA_Z_PNN_model
from loss import SpectralLoss, StructuralLoss

from tools.spectral_tools import gen_mtf, normalize_prisma, denormalize_prisma

from dataset import open_mat
from config_dict import config
from tools.cross_correlation import local_corr_mask
from tools.pca_tools import pca, inverse_pca

from skimage.transform import rescale


def test_pca_z_pnn(args):

    # Paths and env configuration
    basepath = args.input
    method = 'PCA-Z-PNN'
    out_dir = os.path.join(args.out_dir, method)

    gpu_number = args.gpu_number
    use_cpu = args.use_cpu

    # Training hyperparameters

    if args.learning_rate != -1:
        learning_rate = args.learning_rate
    else:
        learning_rate = config['learning_rate']

    # Satellite configuration
    sensor = config['satellite']
    ratio = config['ratio']
    num_blocks = config['num_blocks']
    n_components = config['n_components']
    last_wl = config['last_wl']

    epochs = args.epochs

    if epochs == -1:
        epochs = config['epochs']

    # Environment Configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

    # Devices definition
    device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")

    if sensor == 'PRISMA':
        normalize = normalize_prisma
        denormalize = denormalize_prisma
    else:
        raise 'Satellite not supported'

    # Open the image

    pan, ms_lr, ms, _, wl = open_mat(basepath)

    pan = normalize(pan, nbits=16, nbands=1).to(device)

    criterion_spec = SpectralLoss(gen_mtf(ratio, sensor, kernel_size=61, nbands=n_components), ratio, device).to(device)
    criterion_struct = StructuralLoss(ratio).to(device)

    history_loss_spec = []
    history_loss_struct = []

    alpha = config['alpha_1']

    fused = []

    band_blocks = []

    band_rgb = 0
    while wl[band_rgb] < last_wl:
        band_rgb += 1

    band_blocks.append(ms_lr[:, :band_rgb + 1, :, :])
    band_blocks.append(ms_lr[:, band_rgb:, :, :])

    for block_index in range(num_blocks):

        net = PCA_Z_PNN_model(nbands=n_components).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(config['beta_1'], config['beta_2']))
        net.train()

        ms_lr_pca, W, mu = pca(band_blocks[block_index])

        ms_pca = torch.tensor(rescale(torch.squeeze(ms_lr_pca).numpy(), ratio, order=3, channel_axis=0))[None, :, :, :]
        spec_ref_exp = normalize(ms_pca[:, :n_components, :, :], nbands=ms_pca.shape[1], nbits=16).to(device)
        spec_ref = normalize(ms_lr_pca[:, :n_components, :, :], nbands=ms_pca.shape[1], nbits=16).to(device)

        min_loss = torch.inf

        inp = torch.cat([spec_ref_exp, pan], dim=1)

        threshold = local_corr_mask(inp, ratio, sensor, device, config['semi_width'])

        if block_index == 1:
            alpha = config['alpha_2']

        print('Block index {} / {}'.format(block_index + 1, num_blocks))

        pbar = tqdm(range(epochs))

        for epoch in pbar:

            pbar.set_description('Epoch %d/%d' % (epoch + 1, epochs))

            net.train()
            optim.zero_grad()

            outputs = net(inp)

            loss_spec = criterion_spec(outputs, spec_ref)
            loss_struct, loss_struct_without_threshold = criterion_struct(outputs[:,:1,:,:], pan, threshold[:,:1,:,:])

            loss = loss_spec + alpha * loss_struct

            loss.backward()
            optim.step()

            running_loss_spec = loss_spec.item()
            running_loss_struct = loss_struct_without_threshold

            history_loss_spec.append(running_loss_spec)
            history_loss_struct.append(running_loss_struct)

            if loss.item() < min_loss:
                min_loss = loss.item()
                if not os.path.exists('temp'):
                    os.makedirs(os.path.join('temp'))
                torch.save(net.state_dict(), os.path.join('temp', 'PCA-Z-PNN_best_model.tar'))

            pbar.set_postfix(
                {'Spec Loss': running_loss_spec, 'Struct Loss': running_loss_struct})

        net.eval()
        net.load_state_dict(torch.load(os.path.join('temp', 'PCA-Z-PNN_best_model.tar')))

        ms_pca[:, :n_components, :, :] = denormalize(net(inp), nbands=ms_pca.shape[1], nbits=16)
        fused_block = inverse_pca(ms_pca, W, mu)

        if block_index == 0:
            fused.append(fused_block[:, :-1, :, :].detach().cpu())
        else:
            fused.append(fused_block.detach().cpu())

    fused = torch.cat(fused, 1)
    fused = np.moveaxis(torch.squeeze(fused, 0).numpy(), 0, -1)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_path = os.path.join(out_dir, basepath.split(os.sep)[-1].split('.')[0] + '_PCA-Z-PNN.mat')
    io.savemat(save_path, {'I_MS': fused})
    history = {'loss_spec': history_loss_spec, 'loss_struct': history_loss_struct}
    io.savemat(os.path.join(out_dir, basepath.split(os.sep)[-1].split('.')[0] + '_PCA-Z-PNN_stats.mat'), history)

    torch.cuda.empty_cache()
    gc.collect()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='R-PNN Training code',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='R-PNN is an unsupervised deep learning-based pansharpening '
                                                 'method.',
                                     epilog='''\
Reference: 
PCA-CNN Hybrid Approach for Hyperspectral Pansharpening
G. Guarino, M. Ciotola, G. Vivone, G. Poggi, G. Scarpa 

Authors: 
- Image Processing Research Group of University of Naples Federico II ('GRIP-UNINA')
- National Research Council, Institute of Methodologies for Environmental Analysis (CNR-IMAA)
- University of Naples Parthenope

For further information, please contact the first author by email: giuseppe.guarino2[at]unina.it '''
                                     )
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required named arguments')

    required.add_argument("-i", "--input", type=str, required=True,
                          help='The path of the .mat file'
                               'For more details, please refer to the GitHub documentation.')

    optional.add_argument("-o", "--out_dir", type=str, default='Outputs',
                          help='The directory in which save the outcome.')

    optional.add_argument('-n_gpu', "--gpu_number", type=int, default=0, help='Number of the GPU on which perform the '
                                                                              'algorithm.')
    optional.add_argument("--use_cpu", action="store_true",
                          help='Force the system to use CPU instead of GPU. It could solve OOM problems, but the '
                               'algorithm will be slower.')

    optional.add_argument("-lr", "--learning_rate", type=float, default=-1.0,
                          help='Learning rate with which perform the training.')

    optional.add_argument("--epochs", type=int, default=-1, help='Number of the epochs with which perform the '
                                                                 'target-adaptation of the algorithm.')

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    test_pca_z_pnn(arguments)
