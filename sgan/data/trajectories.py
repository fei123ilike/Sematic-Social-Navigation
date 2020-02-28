
from __future__ import print_function, division
import logging
import os
import math

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, goals_list, goals_rel_list,img_list, pointer_to_image) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    goals = torch.cat(goals_list, dim=0).permute(2, 0, 1)
    goals_rel = torch.cat(goals_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, goals, goals_rel, img_list,pointer_to_image
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def pad_to_square(img, pad_value=0):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def load_images(image_file,scaled_size):
    
    # Extract image as PyTorch tensor then normalize it 
    img = transforms.ToTensor()(Image.open(image_file).convert('RGB'))
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(img)
    
    print("loaded image:",image_file)
    # Handle images with less than three channels
    if len(img.shape) != 3:
        img = img.unsqueeze(0)
        img = img.expand((3, img.shape[1:]))

    _, original_h, original_w = img.shape

    # Pad to square resolution
    img, pad = pad_to_square(img, 0)
    _, padded_h,padded_w = img.shape

    # Resize
    img = resize(img, scaled_size)
    _, resized_h, resized_w = img.shape
    ratio =  resized_h / padded_h
    print("resized channel,resized_h, resized_w and ratio:",img.shape, ratio)
    
    return img, pad, ratio

def scale_peds(origin_peds, pad, ratio):
    """ Scales peds acrroding to scaled image shape"""
    # Adjust for added padding
    origin_peds[:, 2] += pad[0]
    origin_peds[:, 3] += pad[2]
    # adjust for resize
    origin_peds[:, 2] *= ratio
    origin_peds[:, 3] *= ratio

    return origin_peds

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, dset_name, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t', scaled_size=512, cropped_size=512):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.dset_name = dset_name
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.scaled_size = scaled_size

        data_dir = "/home/asus/SemanticSocialNavigation/datasets"
        data_dir = os.path.join(data_dir,dset_name)
      
        all_folders = os.listdir(data_dir)
        all_subfolders = [os.path.join(data_dir, _path) for _path in all_folders if _path[-6:-1] == "video"]
        all_subfolders.sort()
    
        all_files = []
        for _path in all_subfolders:
            filenames  = os.listdir(_path)
            for fn in filenames:
                all_files.append(os.path.join(_path, fn)) 

        image_files = [_path for _path in all_files if _path[-13:] == "reference.jpg"]
        image_files.sort()

        txt_files = [_path for _path in all_files if _path[-13:] == "processed.txt"]
        txt_files.sort()

        
        img_list = []
        pad_list = []
        ratio_list = []
        for image_file in image_files:
            img, pad, ratio = load_images(image_file,self.scaled_size)
            img_list.append(img)
            pad_list.append(pad)
            ratio_list.append(ratio)
            
        self.img_list = img_list 
        
        # process the trajectory
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        goal_list = []
        goal_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        pointer_to_image = []
        
        for image_label, path in enumerate(txt_files):
            data = read_file(path, delim)
            # scale the coordinate x and y accordingly 
            pad = pad_list[image_label]
            ratio = ratio_list[image_label]
            
            

            data = scale_peds(data, pad, ratio)
            # remove the catergory in last column for now
            data = data[:,:-1]
            # Zero padding, the dimension will be used to store exit point and corresponding image label
            data = np.pad(data, [(0, 0), (0, 2)], mode='constant')
            # Inverse tracing to find the exit point of each ped
            exit_point_map = {}
            for i in range(len(data)-1, -1, -1):
                if data[i][1] not in exit_point_map:
                    exit_point_map[data[i][1]] = data[i][2:4]
                data[i][4:6] = exit_point_map[data[i][1]]
                

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
                
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
              
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_goal_rel = np.zeros((len(peds_in_curr_seq), 2, 1))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_goal = np.zeros((len(peds_in_curr_seq), 2, 1))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))

                curr_image_label = np.zeros((len(peds_in_curr_seq),
                                             self.seq_len))
        
                num_peds_considered = 0
                _non_linear_ped = []
                
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
    
                    curr_ped_goal = curr_ped_seq[0, -2:].reshape([2, 1])
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:4])
                    curr_ped_seq_start = curr_ped_seq[:, 0].reshape([2, 1])
                    # Make coordinates relative
                    curr_ped_goal_rel = curr_ped_goal - curr_ped_seq_start
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    # Relative coordinate should base on our position at first timestep
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq_start
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_goal[_idx, :, pad_front:pad_end] = curr_ped_goal
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_goal_rel[_idx, :, pad_front:pad_end] = curr_ped_goal_rel
                   
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    curr_image_label[_idx, pad_front:pad_end] = image_label                        
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    goal_list.append(curr_goal[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    goal_list_rel.append(curr_goal_rel[:num_peds_considered])
                    pointer_to_image.append(curr_image_label[:num_peds_considered])
                   

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        goal_list = np.concatenate(goal_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        goal_list_rel = np.concatenate(goal_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        pointer_to_image = np.concatenate(pointer_to_image, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)


        self.goals = torch.from_numpy(goal_list).type(torch.float)
        self.goals_rel = torch.from_numpy(goal_list_rel).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        
        self.pointer_to_image = torch.from_numpy(pointer_to_image).type(torch.int)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.goals[start:end, :], self.goals_rel[start:end, :],
            self.img_list,
            self.pointer_to_image[start:start+1, :] # only need the pointer of the first person in the scene
        ]
        return out