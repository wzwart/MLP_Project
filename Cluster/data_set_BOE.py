import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import scipy.io
from skimage.transform import resize
from torch.utils.data import Dataset

def thresh(x):
    return (x!=0)*1


class UNetDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, width_in,height_in, width_out, height_out , max_size= None ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        if type (max_size) is type (None):
            self.max_size = -1
        else:
            self.max_size = max_size
        self.input_path = root_dir
        self.width_in=width_in
        self.height_in=height_in
        self.width_out=width_out
        self.height_out=height_out
        self.frac = {"train": (0, 0.7), "valid": (0.7, 0.9), "test": (0.9, 1)}
        self.create_dataset()
        self.for_plotting=False

    def __len__(self):
        return self.length


    def create_dataset(self):
        import pickle
        pickle_path = os.path.join(self.input_path, f"pickle_{self.max_size}.p")
        if os.path.exists(pickle_path):
            print("loading from pickle file")
            data = pickle.load(open(pickle_path, "rb"))
            (self.x, self.y) = data
        else:
            subject_path = [os.path.join(self.input_path, 'Subject_0{}.mat'.format(i)) for i in range(1, 10)] + [
                os.path.join(self.input_path, 'Subject_10.mat')]
            m=len(subject_path)
            x = []
            y = []
            i=0
            data_indexes = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]
            while i < m and (len(x) < self.max_size or self.max_size==-1):
                path = subject_path[i]
                mat = scipy.io.loadmat(path)
                img_tensor = mat['images']
                fluid_tensor = mat['manualFluid1']
                img_array = np.transpose(img_tensor, (2, 0, 1)) / 255
                img_array = resize(img_array, (img_array.shape[0], self.width_in, self.height_in))
                fluid_array = np.transpose(fluid_tensor, (2, 0, 1))
                fluid_array = thresh(fluid_array)
                fluid_array = resize(fluid_array, (fluid_array.shape[0], self.width_out, self.height_out))
                x += [img_array[idx] for idx in data_indexes]
                y += [fluid_array[idx] for idx in data_indexes]
                i+=1

            self.x= np.expand_dims(np.array(x), 1)
            y= np.array(y)
            self.y=np.array([y, 1 - y]).transpose(1, 2, 3, 0)
            data =(self.x, self.y)
            pickle.dump(data, open(pickle_path, "wb"))
        self.length=len(self.x)

    def get_data(self, which_set, for_plotting=False):
        x,y= self.x[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)], self.y[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)]
        return (x,y)

