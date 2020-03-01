import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys
import numpy as np
import tqdm
import scipy.io
from skimage.transform import resize
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
from data_sets.data_set_utils import *



class DatasetYoutubeHM(Dataset):
    """Youtube faces dataset."""

    def __init__(self, root_dir, width_in,height_in, width_out, height_out, num_landmarks, landmarks_collapsed=False,max_size= -1):
        """
        Args:
            root_dir: Directory where the CSV is stored.
            width_in: Width of the input image.
            height_in: Height of the input image.
            width_out: Width of the output heatmap.
            height_out: Height of the output heatmap.
            num_landmarks: Number of landmarks to do the heatmap for.
            landmarks_collapsed: Boolean indicating if all the landmarks should be collapsed to one heatmap.
            max_size: Number of images to consider. If -1, consider all.
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
        self.num_landmarks=num_landmarks
        self.landmarks_collapsed=landmarks_collapsed
        self.frac = {"train": (0, 0.7), "valid": (0.7, 0.9), "test": (0.9, 1)}
        self.create_dataset()

    def __len__(self):
        return self.length


    def create_dataset(self):
        import pickle
        if self.landmarks_collapsed:
            pickle_path = os.path.join(self.input_path, f"pickle_Youtube_{self.max_size}_{self.num_landmarks}_col.p")
        else:
            pickle_path = os.path.join(self.input_path, f"pickle_Youtube_{self.max_size}_{self.num_landmarks}.p")
        if os.path.exists(pickle_path):
            print("loading from pickle file")
            data = pickle.load(open(pickle_path, "rb"))
            (self.x, self.y) = data
        else:
            path = os.path.join(self.input_path,"Youtube.csv")
            dataset = pd.read_csv(path)

            image_paths = dataset.iloc[:, 0].to_numpy()
            landmarks = dataset.iloc[:, 1:].to_numpy()

            for i in range(len(image_paths)):
                image_paths[i] = os.path.join(self.input_path,image_paths[i])

            x = []
            y = []
            if self.max_size==-1:
                number_of_images = len(image_paths)
            else:
                number_of_images = min(len(image_paths),self.max_size)
            n_updates=min(50,number_of_images)
            with tqdm.tqdm(total=n_updates, file=sys.stdout) as pbar_test:  # ini a progress bar
                for i, image_path in enumerate(image_paths):
                    if i%int(number_of_images/n_updates)==0:
                        pbar_test.set_description(
                            f"Generate Images {i} of {number_of_images}")  # update progress bar string output
                        pbar_test.update(1)  # update progress bar status

                    if (self.max_size != -1 and len(x) >= self.max_size):
                        break
                    # Resize the image to the input size
                    resized, points = resizeInput(image_path, landmarks[i], self.width_in, self.height_in)

                    # normalize:
                    resized = resized-resized.mean(axis=(0,1))
                    resized = resized / np.sqrt(resized.var(axis=(0,1)))

                    x.append(resized)

                    # Scale the landmark coordinates to the output size
                    ratio = np.array([(self.width_in / self.width_out), (self.height_in / self.height_out)])
                    points = np.around(points / ratio, decimals=3)
                    # Get the heatmap for each landmark
                    if self.num_landmarks==5:
                        points=points[[37-1,40-1,43-1,46-1,34-1]]

                    # Get the heatmap for each landmark
                    if self.landmarks_collapsed:
                        u = np.zeros((self.width_out, self.height_out, 1))
                    else:
                        u = np.zeros((self.width_out, self.height_out, self.num_landmarks))
                    for j, (x_p, y_p) in enumerate(points[:self.num_landmarks]):
                        if self.landmarks_collapsed:
                            u[:, :, 0] +=generateHeatmap(x_p, y_p, self.width_out, self.height_out)
                        else:
                            u[:, :, j] = generateHeatmap(x_p, y_p, self.width_out, self.height_out)
                    u=np.clip(u,0,1)
                    y.append(u)
            self.x= np.transpose(np.array(x),(0,3,1,2))
            self.y= np.array(y)
            data =(self.x, self.y)
            pickle.dump(data, open(pickle_path, "wb"))
        self.length=len(self.x)

    def get_data(self, which_set):
        return self.x[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)], self.y[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)]

    def render(self, x, y, p, out, number_images):
        from collections import OrderedDict
        from matplotlib import cm

        set1 = cm.get_cmap('Set1')
        colors = np.asarray(set1.colors)
        no_colors = colors.shape[0]
        no_landmarks = y.shape[3]
        if type(out) != type(None):
            no_cols = 2 + no_landmarks
        else:
            no_cols = 2
        fig, ax = plt.subplots(nrows=number_images, ncols=no_cols, figsize=(18, 3 * number_images))
        for row_num in range(number_images):
            x_img = np.transpose(x[row_num], (1, 2, 0))

            x_img = x_img - np.min(x_img, axis=(0, 1))
            x_img = x_img / np.max(x_img, axis=(0, 1))

            y_img = np.array([np.array(
                [y[row_num, :, :, i] * colors[i % no_colors, 0], y[row_num, :, :, i] * colors[i % no_colors, 1],
                 y[row_num, :, :, i] * colors[i % no_colors, 2]]) for i in range(no_landmarks)])
            y_img = np.sum(y_img, axis=0).transpose((1, 2, 0))

            y_img = y_img - np.min(y_img, axis=(0, 1))
            # y_img=y_img/np.max(y_img, axis=(0,1))

            x_img = x_img[:, :, [2, 1, 0]]  # RGB BGR conversion

            ax[row_num][0].imshow(x_img)
            ax[row_num][no_cols - 1].imshow(y_img)
            if type(out) != type(None):
                for i in range(no_landmarks):
                    out_img = (out[row_num] - out[row_num].min())
                    out_img = np.array(
                        [out_img[:, :, i] * colors[i % no_colors, 0], out_img[:, :, i] * colors[i % no_colors, 1],
                         out_img[:, :, i] * colors[i % no_colors, 2]]).transpose((1, 2, 0))
                    ax[row_num][i + 1].imshow(out_img)
        plt.show()