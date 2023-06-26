"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import collections
import csv
import glob
import json
import os
import socket

import numpy as np

from termcolor import cprint
import pandas as pd
import torch

from configs.paths import dataroot
from datasets.base_dataset import BaseDataset
import h5py
from typing import List

hostname = socket.gethostname()
import pandas as pd

import copy

class ScSnetCodeDataset(BaseDataset):

    # def initialize(self, opt, phase='train', cat='chair'):
    def initialize(self, opt, phase='train', cat='chair'):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.phase = phase
        self.cat = cat
        with open(f'./info-shapenet.json') as f:
            self.info = json.load(f)
        
        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        self.model_list = []
        with open(f'./filelists/{self.cat_to_id[cat]}_{phase}.lst') as f:
            lang_list_s = []
            model_list_s = []
            for l in f.readlines():
                model_id = l.rstrip('\n')
                self.model_list.append(model_id)
        all_files = glob.glob(f"../data/chairs/*/*")
        set_zs = [p for p in all_files if "z_set" in p and p.split("/")[-2] in self.model_list]
        shape_zs = [p for p in all_files if "z" in p and "set" not in p and p.split("/")[-2] in self.model_list]

        self.all_zs = np.array(shape_zs)
        self.text2shapepp = pd.read_csv('./similar_phrase_2.csv')
        
        

        
        self.set2path = {p.split("/")[-1].split("_")[-1].replace(".pt", ""): p for p in set_zs}
        self.mod2code_path = {p.split("/")[-2]: p for p in shape_zs}
        # NOTE: set code_root here for transformer_model to load
        # opt.code_dir = self.code_dir
        self.N = len(self.all_zs)
        self.rng = np.random.default_rng(0)
        np.random.shuffle(self.all_zs)


    def __getitem__(self, index):
        try:

            path = self.all_zs[index]
            sdf_path = "/".join(path.split("/")[:-1] + ["ori_sample.h5"])
            z = torch.load(path, map_device="cpu")
            if "set" in path:
                row_id = int(path.split("_")[-1].replace(".pt",""))
                similar_shapes = [self.text2shapepp.iloc[row_id]["model_id"]] + self.text2shapepp.iloc[row_id]["similar_model_ids"]
                shape_selected = random.choice(similar_shapes, 1)
                new_path = copy.deepcopy(path)
                new_path = new_path.split("/")
                new_path[-2] = shape_selected
                new_path[-1] = "z_shape.pt"
                new_path = "/".join(new_path)
                tgt = torch.load(new_path, map_device="cpu")
            else:
                tgt = z
                
            ret = {
                'z_q': z.permute(3,0,1,2),
                'idx': z.permute(3,0,1,2),
                'cat_id': self.cat_to_id[self.cat],
                'cat_str': self.cat,
                'path': sdf_path,
                "tgt": torch.argmax(tgt, axis=-1),                
            }
        except Exception as e:
            print(e)
            return self.__getitem__(self.rng.integers(0, len(self)))

        return ret
    def __len__(self):
        return self.N

    def name(self):
        return 'Text2ShapePP'

