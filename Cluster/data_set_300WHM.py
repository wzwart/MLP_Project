import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import scipy.io
from skimage.transform import resize
from torch.utils.data import Dataset
import cv2

def thresh(x):
    return (x!=0)*1


def resizeInput(image_file, landmarks, width, height):
    ori_image = cv2.imread(image_file)
    height_im, width_im, channels = ori_image.shape
    # resize image
    dim = (width, height)
    resized = cv2.resize(ori_image, dim, interpolation=cv2.INTER_AREA)

    # Modify landmark values

    landmarks = landmarks.astype('float').reshape((int(landmarks.shape[0] / 2), 2))
    ratio = np.array([(width_im / width), (height_im / height)])
    landmarks = landmarks / ratio
    landmarks = np.around(landmarks, decimals=3)

    return resized, landmarks


def bivariateGaussianProb(x, y, meanX, meanY, scale):
    cov = np.eye(2) * 2.5 * (scale / 64)
    x = np.array([x, y])
    # x = xT.reshape((2,1))
    mean = np.array([meanX, meanY])

    diff = x - mean

    const = 1 / (np.sqrt(np.square(2 * np.pi) * np.linalg.det(cov)))

    exp = np.exp((-1 / 2) * np.dot(np.dot(diff, np.linalg.inv(cov)), diff.reshape((2, 1))))

    return const * exp


# Generate heatmap for keypoint (x,y)
def generateHeatmap(x, y, width, height):
    z = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            z[i][j] = bivariateGaussianProb(i, j, x, y, max(width, height))

    return z.T


class Dataset300WHM(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, width_in,height_in, width_out, height_out, num_landmarks, max_size= None):
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
        self.num_landmarks=num_landmarks
        self.frac = {"train": (0, 0.7), "valid": (0.7, 0.9), "test": (0.9, 1)}
        self.create_dataset()

    def __len__(self):
        return self.length


    def create_dataset(self):
        import pickle
        pickle_path = os.path.join(self.input_path, f"pickle_300W_{self.max_size}.p")
        if os.path.exists(pickle_path):
            print("loading from pickle file")
            data = pickle.load(open(pickle_path, "rb"))
            (self.x, self.y) = data
        else:
            path = os.path.join(self.input_path,"300W.csv")
            dataset = pd.read_csv(path)

            image_paths = dataset.iloc[:, 0].to_numpy()
            landmarks = dataset.iloc[:, 1:].to_numpy()

            for i in range(len(image_paths)):
                image_paths[i] = os.path.join(self.input_path,"images/" + image_paths[i])

            x = []
            y = []

            for i, image_path in enumerate(image_paths):
                if (self.max_size != -1 and len(x) >= self.max_size):
                    break
                # Resize the image to the input size
                resized, points = resizeInput(image_path, landmarks[i], self.width_in, self.height_in)
                x.append(resized)

                # Scale the landmark coordinates to the output size
                ratio = np.array([(self.width_in / self.width_out), (self.height_in / self.height_out)])
                points = np.around(points / ratio, decimals=3)
                # Get the heatmap for each landmark
                u = np.zeros((self.width_out, self.height_out, self.num_landmarks))
                for j, (x_p, y_p) in enumerate(points[:self.num_landmarks]):
                    u[:, :, j] = generateHeatmap(x_p, y_p, self.width_out, self.height_out)
                y.append(u)
            self.x= np.array(x)
            self.y= np.array(y)
            data =(self.x, self.y)
            pickle.dump(data, open(pickle_path, "wb"))
        self.length=len(self.x)

    def get_data(self, which_set):
        return self.x[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)], self.y[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)]


