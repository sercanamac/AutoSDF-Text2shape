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
import ast


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
        with open(f'./filelists/{self.cat_to_id[cat]}_train.lst') as f:
            lang_list_s = []
            model_list_s = []
            for l in f.readlines():
                model_id = l.rstrip('\n')
                self.model_list.append(model_id)
        with open(f'./filelists/{self.cat_to_id[cat]}_test.lst') as f:

            for l in f.readlines():
                model_id = l.rstrip('\n')
                self.model_list.append(model_id)
        all_files = glob.glob(f"../shap/*/*")
        set_zs = [p for p in all_files if "z_set" in p and p.split("/")[-2] in self.model_list]
        shape_zs = [p for p in all_files if "z" in p and "set" not in p and p.split("/")[-2] in self.model_list]
        self.set2path = {p.split("/")[-1].split("_")[-1].replace(".pt", ""): p for p in set_zs}
        self.mod2code_path = {p.split("/")[-2]: p for p in shape_zs}
        # NOTE: set code_root here for transformer_model to load
        # opt.code_dir = self.code_dir

        self.text2shapepp = pd.read_csv('./similar_phrase_2.csv')
        with open("file.json", 'r') as f:
            all_id_list = json.load(f)
        self.sequences: List = all_id_list
        seq_to_keep = []
        for seq in self.sequences:
            if self.text2shapepp.iloc[seq[0]]["model_id"] in self.model_list:
                seq_to_keep.append(seq)

        self.sequences = seq_to_keep
        self.N = len(self.sequences)
        self.rng = np.random.default_rng(0)
        self.cache = {}

    def __getitem__(self, index):
        try:

            seq = self.sequences[index]
            t_1_ind = self.rng.integers(low=-1, high=len(seq)-1)
            if t_1_ind == -1:
                t_1_code = torch.fill_(torch.zeros(size=(8,8,8, 512)), value=1/512) 
                t_1_id = ""
                t_1_text = ""
            else:

                t_1_row_ind = seq[t_1_ind]
                t_1_row = self.text2shapepp.iloc[t_1_row_ind]
                t_1_text = t_1_row["phrase_texts"]
                if t_1_row["similar_model_id"] == '0':
                    t_1_id = t_1_row["model_id"]
                    if t_1_id not in self.cache:
                        t_1_code = torch.load(self.mod2code_path[t_1_id], map_location="cpu")
                        self.cache[t_1_id] = t_1_code
                    else:
                        t_1_code = self.cache[t_1_id]
                else:
                    t_1_id = t_1_row["model_id"]
                    if str(t_1_row_ind) not in self.cache:

                        t_1_code = torch.load(self.set2path[str(t_1_row_ind)], map_location="cpu")
                        self.cache[str(t_1_row_ind)] = t_1_code
                    else:
                        t_1_code = self.cache[str(t_1_row_ind)]
            t_2_row_ind = seq[t_1_ind + 1]
            t_2_row = self.text2shapepp.iloc[t_2_row_ind]
            t_2_text = t_2_row["phrase_texts"]
            t_2_row_ind = seq[t_1_ind+1]




            if t_2_row["similar_model_id"] == '0':
                t_2_id = t_2_row["model_id"]
            
                if t_2_id not in self.cache:

                    t_2_code = torch.load(self.mod2code_path[t_2_id], map_location="cpu")
                    self.cache[t_2_id] = t_2_code
                else:
                    t_2_code = self.cache[t_2_id]
                
            else:
                t_2_id = t_2_row["model_id"]

                similar_ids = [t_2_row["model_id"]] + ast.literal_eval(t_2_row["similar_model_id"])
                choosen_one = np.random.choice(similar_ids, 1)[0]
                if str(choosen_one) not in self.cache:
                    
                    t_2_code = torch.load(self.mod2code_path[str(choosen_one)], map_location="cpu")
                    self.cache[str(choosen_one)] = t_2_code
                else:
                    t_2_code = self.cache[str(choosen_one)]
                    
            
            sampler = torch.distributions.categorical.Categorical(t_2_code)
            codeix = sampler.sample()
            t_2_text = t_2_text.replace(t_1_text, "")
            # codeix = t_2_code.permute(3,0,1,2)
            ret = {
                'z_q': t_1_code.float(),
                'idx': codeix,
                'text': t_2_text,
                'text_t1': t_1_text,
                'cat_id': self.cat_to_id[self.cat],
                'cat_str': self.cat,
                'path': t_1_id,
            }
            return ret

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))

    def __len__(self):
        return self.N

    def name(self):
        return 'Text2ShapePP'
