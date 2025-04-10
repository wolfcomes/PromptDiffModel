from itertools import accumulate
import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, npz_path, center=True, transform=None):

        self.transform = transform

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # split data based on mask
        self.data = {}
        for (k, v) in data.items():
            if k == 'names' or k == 'receptors':
                self.data[k] = v
                continue

            # 根据键名选择拆分的依据 mask
            if k.startswith('opt_lig'):
                # 对于优化 ligand 部分，使用 "opt_lig_mask" 来确定样本边界
                mask_key = 'opt_lig_mask'
                sections = np.where(np.diff(data[mask_key]))[0] + 1
            elif 'lig' in k:
                # 对于 reference ligand 部分，使用 "ref_lig_mask" 作为依据
                mask_key = 'ref_lig_mask'
                sections = np.where(np.diff(data[mask_key]))[0] + 1
            elif 'pocket' in k:
                mask_key = 'pocket_mask'
                sections = np.where(np.diff(data[mask_key]))[0] + 1
            else:
                # 如果没有 mask 则不拆分，直接转换为 tensor
                sections = None
            
            if sections is not None:
                self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]
            else:
                self.data[k] = torch.from_numpy(v)

            #self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            if k == 'prompt_labels':
                # 对 prompt_labels 进行样本划分，保证与 lig_coords 保持一致
                sections = np.where(np.diff(data['opt_lig_mask']))[0] + 1
                self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            # 添加每个样本的节点数信息，方便后续使用
            if k == 'ref_lig_mask':
                self.data['num_ref_lig_atoms'] = torch.tensor([len(x) for x in self.data['ref_lig_mask']])
            elif k == 'pocket_mask':
                self.data['num_pocket_nodes'] = torch.tensor([len(x) for x in self.data['pocket_mask']])
            elif k == 'opt_lig_mask':
                self.data['num_opt_lig_atoms'] = torch.tensor([len(x) for x in self.data['opt_lig_mask']])

            # add number of nodes for convenience
            # if k == 'ref_lig_mask':
            #     self.data['num_ref_lig_atoms'] = \
            #         torch.tensor([len(x) for x in self.data['ref_lig_mask']])
            # elif k == 'pocket_mask':
            #     self.data['num_pocket_nodes'] = \
            #         torch.tensor([len(x) for x in self.data['pocket_mask']])

        if center:
            for i in range(len(self.data['opt_lig_coords'])):
                mean = (self.data['ref_lig_coords'][i].sum(0) +
                        self.data['pocket_coords'][i].sum(0) + self.data['opt_lig_coords'][i].sum(0)) / \
                       (len(self.data['ref_lig_coords'][i]) + len(self.data['pocket_coords'][i]) + len(self.data['opt_lig_coords'][i]))
                self.data['ref_lig_coords'][i] = self.data['ref_lig_coords'][i] - mean
                self.data['pocket_coords'][i] = self.data['pocket_coords'][i] - mean
                self.data['opt_lig_coords'][i] = self.data['opt_lig_coords'][i] - mean

    def __len__(self):
        return len(self.data['names'])

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.data.items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():

            if prop == 'names' or prop == 'receptors':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_ref_lig_atoms' or prop == 'num_pocket_nodes' or prop == 'num_opt_lig_atoms'\
                    or prop == 'num_virtual_atoms':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'mask' in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(batch)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out


class ProcessedLigandDataset(Dataset):
    def __init__(self, npz_path, center=True, transform=None):

        self.transform = transform

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # split data based on mask
        self.data = {}
        for (k, v) in data.items():
            if k == 'names' or k == 'receptors':
                self.data[k] = v
                continue

            # 根据键名选择拆分的依据 mask
            if k.startswith('opt_lig'):
                # 对于优化 ligand 部分，使用 "opt_lig_mask" 来确定样本边界
                mask_key = 'opt_lig_mask'
                sections = np.where(np.diff(data[mask_key]))[0] + 1
            elif 'pocket' in k:
                mask_key = 'pocket_mask'
                sections = np.where(np.diff(data[mask_key]))[0] + 1
            else:
                # 如果没有 mask 则不拆分，直接转换为 tensor
                sections = None
            
            if sections is not None:
                self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]
            else:
                self.data[k] = torch.from_numpy(v)

            #self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            if k == 'pocket_mask':
                self.data['num_pocket_nodes'] = torch.tensor([len(x) for x in self.data['pocket_mask']])
            elif k == 'opt_lig_mask':
                self.data['num_opt_lig_atoms'] = torch.tensor([len(x) for x in self.data['opt_lig_mask']])

            # add number of nodes for convenience
            # if k == 'ref_lig_mask':
            #     self.data['num_ref_lig_atoms'] = \
            #         torch.tensor([len(x) for x in self.data['ref_lig_mask']])
            # elif k == 'pocket_mask':
            #     self.data['num_pocket_nodes'] = \
            #         torch.tensor([len(x) for x in self.data['pocket_mask']])

        if center:
            for i in range(len(self.data['opt_lig_coords'])):
                mean = (
                        self.data['pocket_coords'][i].sum(0) + self.data['opt_lig_coords'][i].sum(0)) / \
                       (len(self.data['pocket_coords'][i]) + len(self.data['opt_lig_coords'][i]))
                self.data['pocket_coords'][i] = self.data['pocket_coords'][i] - mean
                self.data['opt_lig_coords'][i] = self.data['opt_lig_coords'][i] - mean

    def __len__(self):
        return len(self.data['names'])

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.data.items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():

            if prop == 'names' or prop == 'receptors':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_pocket_nodes' or prop == 'num_opt_lig_atoms'\
                    or prop == 'num_virtual_atoms':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'mask' in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(batch)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out
