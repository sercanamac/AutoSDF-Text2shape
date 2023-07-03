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

from typing import List

hostname = socket.gethostname()
import time


class Text2ShapePP(BaseDataset):

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

        all_files = glob.glob(f"../raw_dataset/shapenet/*/*")
        #set_zs = [p for p in all_files if "z_set" in p and p.split("/")[-2] in self.model_list]
        #shape_zs = [p for p in all_files if "z" in p and "_set" not in p and p.split("/")[-2] in self.model_list]
        set_zs = [p for p in all_files if "z_set" in p and p.split("/")[-2]]
        shape_zs = [p for p in all_files if "z" in p and "_set" not in p and p.split("/")[-2]]
        self.set2path = {p.split("/")[-1].split("_")[-1].replace(".pt", ""): p for p in set_zs}
        self.mod2code_path = {p.split("/")[-2]: p for p in shape_zs}
        #import pdb;pdb.set_trace()
        # NOTE: set code_root here for transformer_model to load
        # opt.code_dir = self.code_dir
    
        self.text2shapepp = pd.read_csv('../raw_dataset/text2phrase.csv')
        with open("file.json", 'r') as f:
            all_id_list = json.load(f)
        #import pdb;pdb.set_trace()
        self.sequences: List = all_id_list
      
        seq_to_keep = []
        for seq in self.sequences:
            if self.text2shapepp.iloc[seq[0]]["model_id"] in self.model_list:
                seq_to_keep.append(seq)

        self.sequences = seq_to_keep
        self.N = len(self.sequences)
        self.rng = np.random.default_rng(0)
        self.counter = 0


    def __getitem__(self, index):
        try:
            #import pdb;pdb.set_trace()
            seq = self.sequences[index]
            if(len(seq)== 1):
                t_1_ind = 0
                t_1_row_ind = seq[t_1_ind]
                t_2_row_ind = seq[t_1_ind]
            else:
                t_1_ind = self.rng.integers(low=0, high=len(seq)-1)
                t_1_row_ind = seq[t_1_ind]
                t_2_row_ind = seq[t_1_ind+1]
            t_1_row = self.text2shapepp.iloc[t_1_row_ind]
            t_2_row = self.text2shapepp.iloc[t_2_row_ind]
            t_2_text = t_2_row["phrase_texts"]
            t_1_text = t_1_row["phrase_texts"]
            #import pdb;pdb.set_trace()
            t_1_id = t_1_row["model_id"]
            if t_1_row["similar_model_id"] == '0':
                t_1_id = t_1_row["model_id"]
                t_1_code = torch.load(self.mod2code_path[t_1_id], map_location="cpu")
            else:
                t_1_code = torch.load(self.set2path[str(t_1_row_ind)], map_location="cpu")


            if t_2_row["similar_model_id"] == '0':
                t_2_id = t_2_row["model_id"]
                #import pdb;pdb.set_trace()
                similar_ids = [t_2_row["model_id"]]
                choosen_one = np.random.choice(similar_ids, 1)[0]
                #import pdb;pdb.set_trace()
                t_2_code = torch.load(self.mod2code_path[choosen_one], map_location="cpu")
                
            else:
                t_2_code = torch.load(self.set2path[str(t_2_row_ind)], map_location="cpu")
            
            
            sampler = torch.distributions.categorical.Categorical(t_2_code)
            codeix = sampler.sample()
            ret = {
                'z_q': t_1_code,
                'idx': codeix,
                'text': t_2_text,
                'text_t1': t_1_text,
                'cat_id': self.cat_to_id[self.cat],
                'cat_str': self.cat,
                'path': t_1_id,
            }
            
            return ret

        except Exception as ex:
            return self.__getitem__(0)

 
    def __len__(self):
        return self.N

    def name(self):
        return 'Text2ShapePP'

