from torch.utils.data import Dataset
import torch
import os
import h5py
import sys
import os  # noqa



NRRD_EXTENSION = ".nrrd"
OBJ_EXTENSION = ".obj"
PNG_EXTENSION = ".png"
SDF_EXTENSION = ".h5"
DEMO_ARGUMENT = "demo"
SDF_SUFFIX = "_sdf"

def read_sdf(filename, resolution):
    sdf = h5py.File(filename, "r")
    sdf = sdf['pc_sdf_sample'][:]
    sdf_grid = sdf.reshape((resolution, resolution, resolution))
    return sdf_grid



class ShapenetDataset(Dataset):
    def __init__(self, shape_dir, resolution=64, transform=None, max_dataset_size=None, trunc_thres=0.2):
        """Initializes the shape dataset class 

        Args:
            shape_dir (string): directory containing all the shapes
            full_data_set_dir (string): directory containing the text2Shape++ texts
            resolution (int, optional): resolution of sdf grides. Defaults to 64.
            transform (Transform, optional): any transformation applied. Defaults to None.
            cat (str, optional): categories to retrieve all|chairs|tables .Defaults to "all".
        """

        self.shape_dir = shape_dir
        self.transform = transform
        self.resolution = resolution
        self.trunc_thres = trunc_thres
        self.shapes = self.get_directory_ids()
        self.max_dataset_size = max_dataset_size
        if (max_dataset_size):
            self.shapes = self.shapes[0:max_dataset_size]

    def __len__(self):
        return len(self.shapes)

    def get_directory_ids(self):
        directory_name = f"{self.shape_dir}"
        return os.listdir(directory_name)

    def get_shape_directory(self, shape_id):
        return f"{self.shape_dir}/{shape_id}"

    def full_file_path(self, shape_id):
        """Returns the full file path of a given shape id

        Args:
            shape_id (string): The shape id (folder name)

        Returns:
            string: full folder path relative to current working directory
        """
        filename = f"{self.get_shape_directory(shape_id)}/ori_sample{SDF_EXTENSION}"
        # filename = f"{self.shape_dir}/{self.cat}/{shape_id}/{shape_id}{SDF_SUFFIX}{SDF_EXTENSION}"
        return filename

    def get_z_shape(self, shape_id):
        # filename = f"{self.shape_dir}/{self.cat}/{shape_id}/z_shape.pt"
        # tensor = torch.load(filename)
        # return tensor
        try:
            filename = f"{self.get_shape_directory(shape_id)}/{shape_id}/z_shape.pt"
            tensor = torch.load(filename, map_location=torch.device('cpu'))
            return tensor
        except:
            print("Not Found", shape_id)
            zeros = torch.zeros(
                (8, 8, 8, 512))
            zeros[:, :, :, 0] = 1
            return zeros

    def get_item_by_id(self, shape_id):
        """Directly gets the truncated sdf grid of a shape from its id

        Args:
            shape_id (string): ID of the shape being retrieved

        Returns:
           sdf_grid Tenspr[1,resolution,resolution,resolution]: Truncated sdf grid of the shape
        """
        filename = self.full_file_path(shape_id)
        sdf_grid = read_sdf(filename, self.resolution)
        sdf_grid = torch.Tensor(sdf_grid)
        if self.transform:
            sdf_grid = self.transform(sdf_grid)

        thres = self.trunc_thres
        if thres != 0.0:
            sdf_grid = torch.clamp(sdf_grid, min=-thres, max=thres)
        return sdf_grid.view(1, self.resolution, self.resolution, self.resolution)

    def __getitem__(self, idx):
        shape_id = self.shapes[idx]
        return self.get_item_by_id(shape_id)

    def name(self):
        return 'SDFDataset'
