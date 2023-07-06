from datasets.base_dataset import BaseDataset
import json
import re
import pandas as pd
import torch
import numpy as np

TENSOR_EXTENSION = ".pt"
Z_SHAPE_FILENAME = f"z_shape{TENSOR_EXTENSION}"

class Text2Shape(BaseDataset):
    def initialize(self, opt, shape_dir="../raw_dataset", isTrain=True, phase='train', cat='chair'):
        self.opt = opt
        self.shape_dir = shape_dir
        self.max_dataset_size = opt.max_dataset_size
        self.phase = phase
        self.isTrain = isTrain
        with open(f"{shape_dir}/shape_set_paths.json") as f:
              self.shape_set_paths =  json.load(f)
        with open(f"{shape_dir}/text2ShapePP.json") as f2:
              self.sequences = json.loads(json.load(f2))
        #import pdb;pdb.set_trace()
        self.row_indices = np.array(list(self.sequences.keys()))
        self.set_row_indices()
        self.text2shapepp = pd.read_csv(f"{shape_dir}/text2phrase.csv")
        self.counter = 0
        self.counter2 = 0
        self.deleted = ["19c01531fed8ae0b260103f81999a1e1","24cd35785c38c6ccbdf89940ba47dea","30363681727c804095937f6e581cbd41","330d08738230faf0bc78bb6f3ca89e4c", "42e1e9b71b87787fc8687ff9b0b4e4ac","4602fbcc465603882aca5d156344f8d3","4e8d4cffee2c4361c612776a678dd571","6782b941de7b2199a344c33f76676fbd",
"96e9b84ee4a556e8990561fc34164364","bca8d557dac05e082764cfba57a5de73","d764960666572084b1ea4e06e88051f3","d80133e363b9d2c2b5d06744a21b81d8"]
    
    
    def set_row_indices(self):
        if(self.isTrain):
            indices = np.load('datasets/train_indices.npy')
        else:
             indices = np.load('datasets/text2shape_test_indices.npy')
        self.row_indices = self.row_indices[indices.astype(int)]
    
    def get_shape_directory(self, shape_id):
        return f"{self.shape_dir}/shapenet/{shape_id}"
    
    def get_z_set(self, row_index):
        file_name = f"z_set_{row_index}.pt"
        shape_id = self.shape_set_paths.get(file_name, None)
        if(shape_id is None):
            self.counter +=1
            shape_id = self.text2shapepp.iloc[row_index]["model_id"]
            file_name = "z_shape.pt"
        if(shape_id in self.deleted):
            self.counter2+=1
            return torch.full((1,8,8,8,512), 1/512)
        path = f"{self.get_shape_directory(shape_id)}/{file_name}"
        return torch.load(path, map_location="cpu")
        
        
    def get_z_shape(self,row_index):
        shape_id = self.text2shapepp.iloc[row_index]["model_id"]
        if(shape_id in self.deleted):
            return torch.full((1,8,8,8,512), 1/512)
        file_name = "z_shape.pt"
        path = f"{self.get_shape_directory(shape_id)}/{file_name}"
        return torch.load(path, map_location="cpu")    
        
        
    def __getitem__(self, index):
        seq_dict = self.sequences[self.row_indices[index]]
        seq = seq_dict["sequence"]
        current_index = seq_dict["index"]
        previous_index = seq_dict["previous_index"]
        next_index = seq_dict["next_index"]
        
        current_row_index = seq[current_index]
        previous_row_index = seq[previous_index]
        next_row_index = seq[next_index]
        
        current_text = self.text2shapepp.iloc[current_row_index]["phrase_texts"].strip().strip()
        previous_text = self.text2shapepp.iloc[previous_row_index]["phrase_texts"].strip()
        if(current_index != previous_index):
            current_text = current_text.replace(previous_text,'')
            current_text = re.sub(r'[^\w\s]','', current_text).strip()
        if(previous_index==0):
            z_set_prev = torch.full((1,8,8,8,512), 1/512)
        else:
            z_set_prev = self.get_z_set(previous_row_index)
            #z_set_prev = self.get_z_shape(previous_row_index)
        
        z_set_target = self.get_z_set(next_row_index)
        
        return {
            "current_text":current_text,
            "z_set_prev":z_set_prev.float().squeeze(0),
            "z_set_target":z_set_target.float().squeeze(0)
        
        }
        
        
        
 
    def __len__(self):
        length = len(self.row_indices)
        if(self.max_dataset_size is not None and self.max_dataset_size < length):
            return self.max_dataset_size
        return length

    def name(self):
        return 'Text2Shape'