import os.path as osp
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class OldBaseDataset(Dataset):
    def __init__(self, cfgs, split, transform=None):
        self.cfgs = cfgs
        self.bag_info = None
        self.data_path = None
        self.data_root_path = "data/camlyon16/features/npy_files"
        self.k = cfgs["dataset"]["k"]

        self.make_bag(split, pd.read_csv("data/camlyon16/split.csv"))
    
    def make_bag(self, split, bag_info):
        slide_id = bag_info[split].dropna()
        label = bag_info[f"{split}_label"].dropna()
        _bag_info = {}
        data_path = {}
        index = 0
        for _index in range(len(slide_id)):
            _slide_id = slide_id[_index]
            if _slide_id in ["normal_001", "normal_071", "normal_092", "normal_061", "normal_011"]:
                continue
            _label = label[_index]
            _label = int(_label)
            data_path[_slide_id] = osp.join(self.data_root_path, f"{_slide_id}.pt")
            feature = torch.load(data_path[_slide_id])
            _bag_info[index] = {"slide_name": _slide_id, "label": _label, "num_instance": feature.size(0)}
            index += 1
            # self.targets.append(_label)

        self.bag_info = _bag_info
        self.data_path = data_path

    def make_instance_set(self):
        self.instance_info = {}
        count = 0
        for index in range(len(self.bag_info)):
            slide_id = self.bag_info[index]["slide_name"]
            target = self.bag_info[index]["label"]
            num_instance = self.bag_info[index]["num_instance"]
            for instance_index in range(num_instance):
                self.instance_info[count] = {"slide_id": slide_id, "label": target, "instance_index": instance_index}
                count += 1
            print(f'\r{(index+1)/len(self.bag_info)*100:.2f}%', end='', flush=True)
        print(f"\t{count} tiles is collected, {count/len(self.bag_info):.1f} pre slide")
    
    def set_mode(self, mode='bag'):
        self._bag_or_instance = mode
        if mode == 'bag':
            self.data_info = self.bag_info
        elif mode == 'instance':
            self.data_info = self.instance_info
        elif mode == 'selected_bag':
            self.data_info = self.selected_bag_info
        elif mode == 'selected_instance':
            self.data_info = self.selected_instance_info
        else:
            raise ValueError("wrong mode")
    
    def top_k_select(self, pred, is_in_bag=False, inference=False):
        count = 0
        _count = 0
        # shuffle data
        # instance
        index_selected_instance = np.arange(0, len(self.bag_info)*self.k)
        np.random.shuffle(index_selected_instance)
        # bag
        index_selected_bag = np.arange(0, len(self.bag_info))
        np.random.shuffle(index_selected_bag)
        # selection
        for index in range(len(self.bag_info)):
            slide_id = self.bag_info[index]["slide_name"]
            target = self.bag_info[index]["label"]
            num_instance = self.bag_info[index]["num_instance"]
            if inference:
                sigmoid = lambda x: 1/(1 + np.exp(-x))
                _pred_ = sigmoid(pred[count: count+num_instance]).mean(axis=-2)
                _pred = pred[count: count+num_instance, _pred_.argmax(axis=-1)]
            else:
                _pred = pred[count: count+num_instance, 1]
            _index = np.argsort(_pred)  # min, ..., max
            if is_in_bag:
                # k = self.k if self.k < _index.shape[0] else _index.shape[0]
                k = self.k
                self.selected_bag_info[index_selected_bag[_count]] = {"slide_name": slide_id, "label": target, "instance_index": _index[-k:]}
                _count += 1
            else:
                for ii in range(self.k):
                    self.selected_instance_info[index_selected_instance[_count]] = {"slide_name": slide_id, "label": target, "instance_index": _index[-ii]}
                    _count += 1
            count += num_instance
   
    def __getitem__(self, index):
        slide_name = self.data_info[index]["slide_name"]
        target = self.data_info[index]["label"]
        sample = torch.load(self.bag_list[slide_name])
        num_instance = sample.size(0)
        sample = sample.reshape([-1, 1024])
        feature = sample[:num_instance, :]
        feature_p = sample[num_instance:num_instance*2, :]
        feature_m = sample[num_instance*2:, :]

        if self._bag_or_instance == 'bag':
            pass
        elif self._bag_or_instance == 'selected_bag':
            _index = self.data_info[index]["instance_index"]
            _feature = torch.zeros([self.k, 1024])
            for i in range(len(_index)):
                _feature[i] = feature[_index[i]]
            feature = _feature
        else:
            feature = feature[self.data_info[index]["instance_index"]]

        return feature, target, slide_name
    
    def __len__(self):
        return len(self.data_info)


class BagDataset(Dataset):
    def __init__(self, cfgs, split, transform=None):
        self.cfgs = cfgs
        self.bag_name2idx = {}
        self.data_info = None
        self.data_path = None
        self.data_root_path = cfgs["dataset"]["data_path"]

        self.make_bag(split, pd.read_csv(self.cfgs["dataset"]["csv_path"]+f"/{self.cfgs['dataset']['fold']}.csv"))
    
    def make_bag(self, split, bag_info):
        slide_id = bag_info[split].dropna().reset_index(drop=True)
        label = bag_info[f"{split}_label"].dropna().reset_index(drop=True)
        _bag_info = {}
        data_path = {}
        index = 0
        for _index in range(len(slide_id)):
            _slide_id = slide_id[_index]
            _label = label[_index]
            _label = int(_label)
            data_path[_slide_id] = osp.join(self.data_root_path, f"{_slide_id}.pt")
            feature = torch.load(data_path[_slide_id])
            _bag_info[index] = {"slide_name": _slide_id, "label": _label, "num_instance": feature.size(0)}
            self.bag_name2idx[_slide_id] = index
            index += 1

        self.data_info = _bag_info
        self.data_path = data_path

    def __getitem__(self, index):
        slide_name = self.data_info[index]["slide_name"]
        label = self.data_info[index]["label"]
        data = torch.load(self.data_path[slide_name])
        
        feature = data

        return feature, label, slide_name
    
    def __len__(self):
        return len(self.data_info)


class InstanceMILDataset(BagDataset):
    def __init__(self, cfgs, split, transform=None):
        super().__init__(cfgs, split, transform)
        self.bag_info = self.data_info
        self.instance = None
        self.mode = 'bag'
    
    def select_instance(self, prob):
        batch_size = self.cfgs["dataset"]["batch_size"]
        k = self.cfgs["dataset"]["k"]
        instance_data = np.zeros([1, self.cfgs["model"]["input_dim"]+1])
        for i in range(len(self.bag_info)):
            _bag_info = self.bag_info[i]
            _slide_name = _bag_info["slide_name"] 
            _label = _bag_info["label"]
            _prob = prob[_slide_name]

            feature = torch.load(self.data_path[_slide_name])# ["feature"]
            sort_index = np.argsort(_prob[:, 1], axis=-1)[-k:] # min to max
            select_feature = feature[sort_index].numpy()
            select_label = np.ones([k, 1]) * _label
            _instance_data = np.concatenate([select_label, select_feature], axis=1)
            instance_data = np.concatenate([instance_data, _instance_data], axis=0)
        instance_data = instance_data[1:]
        # make dataset
        np.random.shuffle(instance_data)
        num_sample = instance_data.shape[0]
        batch = num_sample // batch_size
        self.instance = {}
        for b in range(batch):
            self.instance[b] = instance_data[b*batch_size:(b+1)*batch_size]
        if num_sample % batch_size != 0:
            self.instance[batch] = instance_data[batch*batch_size:]
        print("[Train] feature select Done!")
    
    def set_mode(self, mode="bag"):
        self.mode = mode
        if mode == "bag":
            self.data_info = self.bag_info
        elif mode == "instance":
            self.data_info = self.instance
        else:
            Warning("wrong dataset mode!")

    def __getitem__(self, index):
        if self.mode == "bag":
            slide_name = self.data_info[index]["slide_name"]
            label = self.data_info[index]["label"]
            data = torch.load(self.data_path[slide_name])
            
            feature = data

            return feature, label, slide_name
        elif self.mode == "instance":
            data = self.data_info[index]
            label = torch.from_numpy(data[:, 0]).long()
            feature = torch.from_numpy(data[:, 1:]).float()

            return feature, label
        else:
            Warning("wrong dataset mode!")
    
    def __len__(self):
        return len(self.data_info)