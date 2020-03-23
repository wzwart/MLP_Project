import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys
import numpy as np
import tqdm
import scipy.io
from skimage.transform import resize
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import torch
import torch.nn.functional as F
from data_sets.data_set_utils import *
import pickle


class Dataset_300W_YT(Dataset):
    """300W and Youtube Faces datasets"""

    def __init__(self, root_dir, width_in, height_in, width_out, height_out, num_landmarks, rbf_width, which_dataset, force_new_pickle, save_pickle, experiment, test_dataset, landmarks_collapsed=False, max_size= -1):
        """
        Args:
            root_dir: Data directory containing both 300W and Youtube Faces directories.
            width_in: Width of the input image.
            height_in: Height of the input image.
            width_out: Width of the output heatmap.
            height_out: Height of the output heatmap.
            num_landmarks: Number of landmarks to do the heatmap for.
            which_dataset: Use 300W (0), Youtube faces (1), both (2), helen + Youtube in training and 300W in test (3)
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
        self.rbf_width=rbf_width
        self.which_dataset=which_dataset
        self.force_new_pickle=force_new_pickle
        self.save_pickle=save_pickle
        self.landmarks_collapsed=landmarks_collapsed
        self.frac = {"train": (0, 0.7), "valid": (0.7, 0.9), "test": (0.9, 1)}
        self.test_dataset = test_dataset
        self.create_dataset()
        self.experiment = experiment

    def __len__(self):
        return self.length

    def set_pickle_path(self):
        names_lookup={0:"300W", 1:"Youtube", 2:"300W_Youtube", 3:"Benchmark"}
        self.pickle_path = os.path.join(self.input_path, f"pickle_{names_lookup[self.which_dataset]}_{self.max_size}_{self.num_landmarks}_XX.p")
        if self.landmarks_collapsed:
            self.pickle_path=self.pickle_path.replace(".p","_col.p")

    def pickle_save(self, x,y,p,norm):
        from math import ceil
        self.set_pickle_path()
        import string
        S= string.ascii_uppercase
        n=x.shape[0]
        n_chunks = 5
        chunk_size = int(ceil(n/n_chunks))
        path_names = [self.pickle_path.replace("XX", S[i]) for i in range(n_chunks)]
        for i in range(n_chunks):
            first_idx = i*chunk_size
            last_idx = min((i+1)*chunk_size,n+1)
            pickle.dump((x[first_idx:last_idx], y[first_idx:last_idx], p[first_idx:last_idx], norm[first_idx:last_idx]), open(path_names[i], "wb"))

    def pickle_load(self):

        self.set_pickle_path()
        import glob
        files = glob.glob(self.pickle_path.replace("XX", '*'))
        if self.force_new_pickle:
            for file in files:
                os.remove(file)
            return False, None
        elif len(files)==0:
            return False, None
        else:
            x=[]
            y=[]
            p=[]
            n=[]
            for file in files:
                (x_chunk,y_chunk,p_chunk,n_chunk) = pickle.load(open(file, "rb"))
                x.append(x_chunk)
                y.append(y_chunk)
                p.append(p_chunk)
                n.append(n_chunk)
            x = np.vstack(x)
            y = np.vstack(y)
            p = np.vstack(p)
            n = np.hstack(n)

            return True, (x,y,p,n)


    def create_dataset(self):
        pickle_load_success , data =  self.pickle_load()
        if pickle_load_success:
            (self.x, self.y, self.p, self.n) = data
        else:
            if (self.which_dataset == 0):
                path = os.path.join(self.input_path,"300W/300W.csv")

                dataset = pd.read_csv(path)
                image_paths = dataset.iloc[:, 0].to_numpy()
                landmarks = dataset.iloc[:, 1:].to_numpy()

                for i in range(len(image_paths)):
                    image_paths[i] = os.path.join(self.input_path, "300W/images/" + image_paths[i])

            elif (self.which_dataset == 1):

                path = os.path.join(self.input_path, "Youtube/Youtube.csv")

                dataset = pd.read_csv(path)
                image_paths = dataset.iloc[:, 0].to_numpy()
                landmarks = dataset.iloc[:, 1:].to_numpy()

                for i in range(len(image_paths)):
                    image_paths[i] = os.path.join(self.input_path, "Youtube/" + image_paths[i])

            elif (self.which_dataset == 2):
                path1 = os.path.join(self.input_path, "300W/300W.csv")
                path2 = os.path.join(self.input_path,"Youtube/Youtube.csv")

                dataset1 = pd.read_csv(path1)
                dataset2 = pd.read_csv(path2)

                image_paths1 = dataset1.iloc[:, 0].to_numpy()
                image_paths2 = dataset2.iloc[:, 0].to_numpy()

                for i in range(len(image_paths1)):
                    image_paths1[i] = os.path.join(self.input_path, "300W/images/" + image_paths1[i])

                for i in range(len(image_paths2)):
                    image_paths2[i] = os.path.join(self.input_path, "Youtube/" + image_paths2[i])

                image_paths = np.hstack((image_paths1, image_paths2))

                landmarks1 = dataset1.iloc[:, 1:].to_numpy()
                landmarks2 = dataset2.iloc[:, 1:].to_numpy()

                landmarks = np.vstack((landmarks1, landmarks2))

            elif (self.which_dataset == 3):
                path1 = os.path.join(self.input_path, "helen/helen.csv")
                path2 = os.path.join(self.input_path,"Youtube/Youtube.csv")

                dataset1 = pd.read_csv(path1)
                dataset2 = pd.read_csv(path2)

                image_paths1 = dataset1.iloc[:, 0].to_numpy()
                image_paths2 = dataset2.iloc[:, 0].to_numpy()

                for i in range(len(image_paths1)):
                    image_paths1[i] = os.path.join(self.input_path, "helen/images/" + image_paths1[i])

                for i in range(len(image_paths2)):
                    image_paths2[i] = os.path.join(self.input_path, "Youtube/" + image_paths2[i])

                image_paths = np.hstack((image_paths1, image_paths2))

                landmarks1 = dataset1.iloc[:, 1:].to_numpy()
                landmarks2 = dataset2.iloc[:, 1:].to_numpy()

                landmarks = np.vstack((landmarks1, landmarks2))

            image_paths, landmarks = shuffle(image_paths, landmarks, random_state=0)
            self.x = []
            self.y = []
            self.p = []
            self.n = []
            if self.max_size == -1:
                number_of_images = len(image_paths)
            else:
                number_of_images = min(len(image_paths),self.max_size)
            n_updates = min(50, number_of_images)
            with tqdm.tqdm(total=n_updates , file=sys.stdout) as pbar_test:  # ini a progress bar
                for i, image_path in enumerate(image_paths):
                    if i % (number_of_images//n_updates) == 0:
                        pbar_test.set_description(
                            f"Generate Images {i} of {number_of_images}")  # update progress bar string output
                        pbar_test.update(1)  # update progress bar status

                    if (self.max_size != -1 and len(self.x) >= self.max_size):
                        break
                    # Resize the image to the input size
                    resized, points = resizeInput(image_path, landmarks[i], self.width_in, self.height_in)

                    # normalize:
                    resized = resized-resized.mean(axis=(0,1))
                    resized = resized / np.sqrt(resized.var(axis=(0,1)))

                    self.x.append(resized)

                    # Scale the landmark coordinates to the output size
                    ratio = np.array([(self.width_in / self.width_out), (self.height_in / self.height_out)])
                    #points = np.around(points / ratio)
                    points = points / ratio
                    #Distance between corners of the eye
                    crnr_eyes = np.sqrt(np.sum(np.square(points[45] - points[36])))

                    #Distance between eye centres
                    left_centre = (points[39] + points[36])/2
                    right_centre = (points[45] + points[42]) / 2
                    ctr_eyes = np.sqrt(np.sum(np.square(right_centre - left_centre)))

                    #Bounding box normalisation
                    sqrt_xy = np.sqrt(self.width_out*self.height_out)

                    norm_dict = {"crnr_eyes": crnr_eyes, "ctr_eyes": ctr_eyes, "sqrt_xy": sqrt_xy}

                    self.n.append(norm_dict)

                    # Get the heatmap for each landmark
                    if self.num_landmarks==5:
                        points=points[[37-1,40-1,43-1,46-1,34-1]]

                    if self.landmarks_collapsed:
                        u = np.zeros((self.width_out, self.height_out, 1))
                    else:
                        u = np.zeros((self.width_out, self.height_out, self.num_landmarks))
                    for j, (x_p, y_p) in enumerate(points[:self.num_landmarks]):
                        if self.landmarks_collapsed:
                            u[:, :, 0] +=generateHeatmap(x_p, y_p, self.width_out, self.height_out, self.rbf_width)
                        else:
                            u[:, :, j] = generateHeatmap(x_p, y_p, self.width_out, self.height_out, self.rbf_width)
                    u=np.clip(u,0,1)
                    self.y.append(u)
                    self.p.append(points[:self.num_landmarks])


            if (self.which_dataset == 1 or self.which_dataset == 3):
                path = os.path.join(self.input_path, "300W/300W.csv")

                dataset = pd.read_csv(path)
                image_paths = dataset.iloc[:, 0].to_numpy()
                landmarks = dataset.iloc[:, 1:].to_numpy()

                test_paths = []
                test_landmarks = []

                for i in range(len(image_paths)):
                    if(image_paths[i].split("_")[0] == self.test_dataset):
                        test_paths.append(os.path.join(self.input_path, "300W/images/" + image_paths[i]))
                        test_landmarks.append(landmarks[i])

                number_of_images = len(test_paths)
                n_updates = min(50, number_of_images)
                with tqdm.tqdm(total=n_updates, file=sys.stdout) as pbar_test:  # ini a progress bar
                    for i, image_path in enumerate(test_paths):
                        if i % (number_of_images // n_updates) == 0:
                            pbar_test.set_description(
                                f"Generate Test Images {i} of {number_of_images}")  # update progress bar string output
                            pbar_test.update(1)  # update progress bar status

                        # Resize the image to the input size
                        resized, points = resizeInput(image_path, test_landmarks[i], self.width_in, self.height_in)

                        # normalize:
                        resized = resized - resized.mean(axis=(0, 1))
                        resized = resized / np.sqrt(resized.var(axis=(0, 1)))

                        self.x.append(resized)

                        # Scale the landmark coordinates to the output size
                        ratio = np.array([(self.width_in / self.width_out), (self.height_in / self.height_out)])
                        # points = np.around(points / ratio)
                        points = points / ratio
                        # Distance between corners of the eye
                        crnr_eyes = np.sqrt(np.sum(np.square(points[45] - points[36])))

                        # Distance between eye centres
                        left_centre = (points[39] + points[36]) / 2
                        right_centre = (points[45] + points[42]) / 2
                        ctr_eyes = np.sqrt(np.sum(np.square(right_centre - left_centre)))

                        # Bounding box normalisation
                        sqrt_xy = np.sqrt(self.width_out * self.height_out)

                        norm_dict = {"crnr_eyes": crnr_eyes, "ctr_eyes": ctr_eyes, "sqrt_xy": sqrt_xy}

                        self.n.append(norm_dict)

                        # Get the heatmap for each landmark
                        if self.num_landmarks == 5:
                            points = points[[37 - 1, 40 - 1, 43 - 1, 46 - 1, 34 - 1]]

                        if self.landmarks_collapsed:
                            u = np.zeros((self.width_out, self.height_out, 1))
                        else:
                            u = np.zeros((self.width_out, self.height_out, self.num_landmarks))
                        for j, (x_p, y_p) in enumerate(points[:self.num_landmarks]):
                            if self.landmarks_collapsed:
                                u[:, :, 0] += generateHeatmap(x_p, y_p, self.width_out, self.height_out, self.rbf_width)
                            else:
                                u[:, :, j] = generateHeatmap(x_p, y_p, self.width_out, self.height_out, self.rbf_width)
                        u = np.clip(u, 0, 1)
                        self.y.append(u)
                        self.p.append(points[:self.num_landmarks])

            self.x = np.transpose(np.array(self.x), (0, 3, 1, 2))
            self.y = np.array(self.y)
            self.p = np.array(self.p)
            self.n = np.array(self.n)

            if(self.save_pickle):
                self.pickle_save(self.x,self.y,self.p,self.n)

        self.length=len(self.x)

    def get_data(self, which_set):

        if (self.which_dataset != 1):
            return self.x[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)], self.y[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)], self.p[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)], self.n[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)]

        else:
            if(which_set == 'train'):
                return self.x[0:int(0.8 * (self.length-300))], self.y[0:int(0.8*(self.length-300))], self.p[0:int(0.8 * (self.length-300))], self.n[0:int(0.8*(self.length-300))]
            elif (which_set == 'valid'):
                return self.x[int(0.8*(self.length-300)):int(self.length-300)], self.y[int(0.8*(self.length-300)):int(self.length-300)], self.p[int(0.8*(self.length-300)):int(self.length-300)], self.n[int(0.8*(self.length-300)):int(self.length-300)]
            elif (which_set == 'test'):
                return self.x[int(self.length-300):int(self.length)], self.y[int(self.length-300):int(self.length)], self.p[int(self.length-300):int(self.length)], self.n[int(self.length-300):int(self.length)]




