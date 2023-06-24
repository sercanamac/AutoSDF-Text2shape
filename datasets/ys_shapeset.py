from torch.utils.data import Dataset
import numpy as np
import torch
import os
import json
from datasets.ShapeNetSample import ShapeNetSample
from torch.distributions import Categorical
TENSOR_EXTENSION = ".pt"
Z_SHAPE_FILENAME = f"z_shape{TENSOR_EXTENSION}"

def sample_z_set(z_set):
    """ Transform z_set into a categorical distribution then sample

    Args:
        z_set (tensor[*]): tensor of any dimension represents a probability distribution
                           over the last dimension
    Returns:
        q_set (tensor[z_set.shape]): one hot encoded tensor with the same shape as z_set
    """

    # Transform z_set into a categorical distribution then sample
    # one hot encode for cross entropy loss  g^3 x 512
    num_classes = z_set.shape[-1]
    q_set = Categorical(z_set).sample()
    # q_set = functional.one_hot(q_set, num_classes=num_classes)
    return q_set

class ShapeNetZSets(Dataset):
    def __init__(self, shape_dir="../raw_dataset", shape_set_paths="shape_set_paths.json", cat="chairs", max_data_set_size=None, use_single_shapes_only=False, isTrain=True):
        """Aggregation dataset of all z_shapes and z_sets

        Args:
            shape_dir (string): directory containing all the shapes
            shape_set_paths (dict {shape_set_filename: directory }, optional): dict containing all the shapesets saved . Defaults to "shape_set_paths.json".
            cat (str chairs|tables , optional): category of shapenet shape . Defaults to "chairs".
        """
        json_file = open(f"{shape_dir}/{shape_set_paths}")
        self.shape_set_paths_json = json.load(json_file)
        self.shape_dir = shape_dir
        self.cat = cat
        self.isTrain = isTrain
        del self.shape_set_paths_json["z_set_42181.pt"]
        self.shape_sets = self.get_shape_sets()
        self.single_shapes = self.get_single_shapes()
        self.max_data_size = max_data_set_size
        self.sample_dataset = ShapeNetSample(shape_dir,shape_dir)
        self.use_single_shapes_only = use_single_shapes_only
    
    
    def get_single_shapes(self):
        dirs = os.listdir(f"{self.shape_dir}/shapenet")
        if(self.isTrain):
            return dirs[0:6000]
        
        return dirs[6000:]
    
    def get_shape_sets(self):
        ids = list(self.shape_set_paths_json.keys())
        if(self.isTrain):
            return ids[0:39000]
        return ids[39000:]
    
    
    def __len__(self):
        if (self.max_data_size != None):
            return self.max_data_size
        if(self.use_single_shapes_only):
            return len(self.single_shapes)
        return len(self.shape_sets) + len(self.single_shapes)

    # TODO: Handle this better with the shapenet dataset

    def get_shape_directory(self, shape_id):
        """returns full path of the shape directory belonging to the shape_id

        Args:
            shape_id string: current shape id

        Returns:
            string: full path of the shape directory
        """
        return f"{self.shape_dir}/shapenet/{shape_id}"

    
    def access_file_single(self,idx):
        shape_id = self.single_shapes[idx]
        directory = self.get_shape_directory(shape_id)
        filename = Z_SHAPE_FILENAME
        return f"{directory}/{filename}", f"{directory}/{Z_SHAPE_FILENAME}"
    
    def access_file(self, idx):
        if(self.use_single_shapes_only):
            return self.access_file_single(idx)
        if (idx < len(self.shape_sets)):
            file_name = self.shape_sets[idx]
            z_set_shape_id = self.shape_set_paths_json[file_name]
            directory = self.get_shape_directory(z_set_shape_id)
            return f"{directory}/{file_name}", self.sample_z_set(z_set_shape_id, file_name)
        else:
            adjusted_index = idx - len(self.shape_sets)
            shape_id = self.single_shapes[adjusted_index]
            directory = self.get_shape_directory(shape_id)
            filename = Z_SHAPE_FILENAME
            return f"{directory}/{filename}", f"{directory}/{Z_SHAPE_FILENAME}"
        
    def sample_z_set(self, shape_id, file_name):
            row_index = file_name.split("_")[-1].split(".")[0]
            model_id = self.sample_dataset.sample(shape_id,int(row_index))
            directory =  self.get_shape_directory(shape_id)
            filename = Z_SHAPE_FILENAME
            return f"{directory}/{filename}"
   
    def initialize(opts):
        print("Initialized")
        
    
    def name(self):
        return "OwnDataset"
    
            

    def __getitem__(self, idx):
        """Gets an item from the dataset

        Args:
            idx (int): index of the given element

        Returns:
            {z_set, q_set}: one hot encoded representations g^3 * codebook indices,
                            z_set is the input data and q_set is the target vector  
        """
        file_access_z_set, file_access_q_set = self.access_file(idx)
        #import pdb;pdb.set_trace()
        z_set = torch.load(file_access_z_set, map_location=torch.device('cpu'))
        idx = sample_z_set(z_set)
        q_set = torch.load(file_access_q_set, map_location=torch.device('cpu'))
#         q_set = sample_z_set(z_set)
#         #z_shape = q_set
#         z_shape = torch.load(file_access_z_shape, map_location=torch.device('cpu'))
#         #z_shape_index = sample_z_set(z_shape)


        
        return {
            "z_set": z_set,
            "q_set": q_set,
            "idx": idx,
            "cat_str":"chair",
            "cat_id":"03",
            "z_q": z_set.permute(3,0,1,2)
        }
