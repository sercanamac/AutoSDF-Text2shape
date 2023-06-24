from torch.utils.data import Dataset
from ast import literal_eval
from datasets.ys_shapenet import ShapenetDataset
import pandas as pd
import torch
import numpy as np


class ShapeNetSample():
    def __init__(self, text2Shape_dir, shape_dir, csv_file_name="text2phrase.csv", resolution=64):
        self.shape_dir = shape_dir
        self.text2Shape_dir = text2Shape_dir
        self.shape_dir = shape_dir
        self.shapenet_dataset = ShapenetDataset(
            shape_dir, resolution=resolution)
        self.csv_path = f"{text2Shape_dir}/{csv_file_name}"
        self.shape_dict = self.construct_shape_dict()

    def construct_shape_dict(self):
        df = pd.read_csv(self.csv_path)
        # set rows with the no similar shapes to the empty array
        df.loc[df['similar_model_id'] == "0", "similar_model_id"] = "[]"
        df.loc[df['similar_model_score'] == "0", "similar_model_score"] = "[]"
        # evaluate the the array string representation to actual arrays
        df['similar_model_id'] = df['similar_model_id'].apply(literal_eval)
        df["similar_model_score"] = df['similar_model_score'].apply(
            literal_eval)
        shape_dict = {}
        # Model ids are not unique in the df. pd.todict can't be used here
        for row in df.itertuples(index=False):
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
        item = self.get_item(model_id,row_index)
        index = item['csv_row_indices'].index(row_index)
        similar_models = item['similar_models'][index]
        similar_scores = item['similar_scores'][index]
        ps = np.array(similar_scores)
        ps = np.exp(ps)
        ps /= np.sum(ps)
        model_id = np.random.choice(similar_models,p=ps)
        is_model_there = self.shape_dict.get(model_id, None)
        if(is_model_there is None):
            return "7f647aa4750640fdaf192dd05d69c7a2"
        return model_id