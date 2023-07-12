from torch.utils.data import Dataset
from ast import literal_eval
from datasets.ys_shapenet import ShapenetDataset
import pandas as pd
import torch
import numpy as np


class ShapeNetSample():
    def __init__(self, text2Shape_dir="../raw_dataset", shape_dir="../raw_dataset", csv_file_name="text2phrase.csv", resolution=64):
        self.shape_dir = shape_dir
        self.text2Shape_dir = text2Shape_dir
        self.shape_dir = shape_dir
        self.shapenet_dataset = ShapenetDataset(
            shape_dir, resolution=resolution)
        self.csv_path = f"{text2Shape_dir}/{csv_file_name}"
        self.df = pd.read_csv(self.csv_path)
        self.shape_dict = self.construct_shape_dict()
        self.deleted = ["19c01531fed8ae0b260103f81999a1e1","24cd35785c38c6ccbdf89940ba47dea","30363681727c804095937f6e581cbd41","330d08738230faf0bc78bb6f3ca89e4c", "42e1e9b71b87787fc8687ff9b0b4e4ac","4602fbcc465603882aca5d156344f8d3","4e8d4cffee2c4361c612776a678dd571","6782b941de7b2199a344c33f76676fbd",
"96e9b84ee4a556e8990561fc34164364","bca8d557dac05e082764cfba57a5de73","d764960666572084b1ea4e06e88051f3","d80133e363b9d2c2b5d06744a21b81d8"]


    def construct_shape_dict(self):
        self.df = pd.read_csv(self.csv_path)
        # set rows with the no similar shapes to the empty array
        self.df.loc[self.df['similar_model_id'] == "0", "similar_model_id"] = "[]"
        self.df.loc[self.df['similar_model_score'] == "0", "similar_model_score"] = "[]"
        # evaluate the the array string representation to actual arrays
        self.df['similar_model_id'] = self.df['similar_model_id'].apply(literal_eval)
        self.df["similar_model_score"] = self.df['similar_model_score'].apply(
            literal_eval)
        shape_dict = {}
        # Model ids are not unique in the df. pd.todict can't be used here
        for row in self.df.itertuples(index=False):
            shape_dict.setdefault(
                row.model_id, {"similar_models": [], "similar_scores": [], "csv_row_indices": []})
            # If similar models are found
            if (len(row.similar_model_id) > 0):
                shape_dict[row.model_id]["similar_models"].append(
                    row.similar_model_id)
                shape_dict[row.model_id]["similar_scores"].append(
                    row.similar_model_score)
                shape_dict[row.model_id]["csv_row_indices"].append(row.index)
        # only consider shapes that have sdf in the dataset
        model_ids = shape_dict.keys()
        self.model_ids = list(
            filter(lambda x: x in model_ids, self.shapenet_dataset.shapes))
        return shape_dict
      
    
    def get_item(self, model_id, row_index):
        return self.shape_dict[model_id]
    
    def sample(self,model_id,row_index):
        row = self.df.iloc[row_index]
        similar_models = row["similar_model_id"]
        if len(similar_models)==0:
            return model_id
        similar_scores = row["similar_model_score"]
        ps = np.array(similar_scores)
        ps = np.exp(ps)
        ps /= np.sum(ps)
        #next_model_id = similar_models[np.argmax(ps)]
        next_model_id = np.random.choice(similar_models,p=ps)
        #import pdb;pdb.set_trace()
        if(next_model_id in self.deleted):
            return model_id
        return next_model_id
    
        #import pdb;pdb.set_trace()
#         is_model_there = self.shape_dict.get(model_id, None)
#         if(is_model_there is None):
#             return "7f647aa4750640fdaf192dd05d69c7a2"
#         item = self.get_item(model_id,row_index)
#         import pdb;pdb.set_trace()
#         index = item['csv_row_indices'].index(row_index)
#         similar_models = item['similar_models'][index]
#         similar_scores = item['similar_scores'][index]
#         ps = np.array(similar_scores)
#         ps = np.exp(ps)
#         ps /= np.sum(ps)
#         #model_id = np.random.choice(similar_models,p=ps)
#         model_id = similar_models[np.argmax(ps)]

        return model_id