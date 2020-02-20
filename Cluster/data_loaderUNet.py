import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import pandas as pd
import scipy.io
from skimage.transform import resize
from tqdm import tqdm, trange
from torch.utils.data import Dataset

def thresh(x):
    return (x!=0)*1

class UNetDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, which_set, width_in,height_in, width_out, height_out , max_size= None ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        if type (max_size) is type (None):
            self.max_size = len(self.key_pts_frame)
        else:
            self.max_size = max_size
        self.which_set= which_set
        self.input_path = root_dir
        self.width_in=width_in
        self.height_in=height_in
        self.width_out=width_out
        self.height_out=height_out


    def __len__(self):
        frac = {"train": (0, 0.7), "valid": (0.7, 0.9), "test": (0.9, 1)}

        return self.max_size


    def create_dataset(self, paths, width_in, height_in, width_out, height_out, data_indexes, mat):
        x = []
        y = []
        for path in tqdm(paths):
            mat = scipy.io.loadmat(path)
            img_tensor = mat['images']
            fluid_tensor = mat['manualFluid1']
            img_array = np.transpose(img_tensor, (2, 0, 1)) / 255
            img_array = resize(img_array, (img_array.shape[0], width_in, height_in))
            fluid_array = np.transpose(fluid_tensor, (2, 0, 1))
            fluid_array = thresh(fluid_array)
            fluid_array = resize(fluid_array, (fluid_array.shape[0], width_out, height_out))

            for idx in data_indexes:
                x += [np.expand_dims(img_array[idx], 0)]
                y += [np.expand_dims(fluid_array[idx], 0)]
        return np.array(x), np.array(y)

    def get_data(self):
        frac={"train":(0,0.7),"valid":(0.7,0.9),"test":(0.9,1)}
        subject_path = [os.path.join(self.input_path, 'Subject_0{}.mat'.format(i)) for i in range(1, 10)] + [
            os.path.join(self.input_path, 'Subject_10.mat')]
        m=len(subject_path)
        data_indexes = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]
        mat = scipy.io.loadmat(subject_path[0])
        img_tensor = mat['images']
        manual_fluid_tensor_1 = mat['manualFluid1']
        img_array = np.transpose(img_tensor, (2, 0, 1))
        manual_fluid_array = np.transpose(manual_fluid_tensor_1, (2, 0, 1))
        print(len(subject_path))
        x, y = self.create_dataset(subject_path[int((frac[self.which_set][0]*m)):int((frac[self.which_set][1]*m))], self.width_in, self.height_in, self.width_out, self.height_out,
                                          data_indexes, mat)
        return x,y
