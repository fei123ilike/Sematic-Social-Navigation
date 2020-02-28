import argparse
import os
import torch

import sys
sys.path.append('/home/asus/SocialNavigation/src/sgan/scripts/sgan')

from attrdict import AttrDict
from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples, plot=True):
    ade_outer, fde_outer = [], []
    total_traj = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            count += 1
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))
                # add plot module
                if plot:
                    _plot_dir = '../saves/'
                    if not os.path.exists(_plot_dir):
                        os.makedirs(_plot_dir)

                    fig = plt.figure()
                    
                    whole_traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
                    whole_traj_fake = whole_traj_fake[:, 0, :]
                    whole_traj_gt = torch.cat([obs_traj, pred_traj_gt], dim=0)
                    whole_traj_gt = whole_traj_gt[:, seq_start_end[0][0]:seq_start_end[0][1], :]

                    y_upper_limit = max([torch.max(whole_traj_fake[:, 1]).data, 
                                         torch.max(whole_traj_gt[:, :, 1]).data]) + 1.

                    y_lower_limit = min([torch.min(whole_traj_fake[:, 1]).data, 
                                         torch.min(whole_traj_gt[:, :, 1]).data]) - 1.

                    x_upper_limit = max([torch.max(whole_traj_fake[:, 0]).data, 
                                         torch.max(whole_traj_gt[:, :, 0]).data]) + 1.

                    x_lower_limit = min([torch.min(whole_traj_fake[:, 0]).data, 
                                         torch.min(whole_traj_gt[:, :, 0]).data]) - 1.

                    def plot_time_step(i):
                        fig, ax = plt.subplots()
                        # ax.plot(goal_point[0].cpu().numpy(), goal_point[1].cpu().numpy(), 'gx')
                        # plot last three point
                        gt_points_x = whole_traj_gt[max(i-2, 0):i+1,:,0].cpu().numpy().flatten()
                        gt_points_y = whole_traj_gt[max(i-2, 0):i+1,:,1].cpu().numpy().flatten()
                        ax.plot(gt_points_x, gt_points_y, 'b.')

                        fake_points_x = whole_traj_fake[max(i-2, 0):i+1,0].cpu().numpy()
                        fake_points_y = whole_traj_fake[max(i-2, 0):i+1,1].cpu().numpy()
                        if i >= args.obs_len:
                            ax.plot(fake_points_x, fake_points_y, 'r*')
                        else:
                            ax.plot(fake_points_x, fake_points_y, 'g.')

                        ax.set_ylim(y_lower_limit.cpu(), y_upper_limit.cpu())
                        ax.set_xlim(x_lower_limit.cpu(), x_upper_limit.cpu())

                        fig.canvas.draw()
                        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        plt.close(fig)

                        return image
                        

                    imageio.mimsave(_plot_dir+str(count)+'.gif', 
                                    [plot_time_step(i) for i in range(args.obs_len+args.pred_len)],
                                    fps=2)

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
