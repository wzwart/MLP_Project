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
import matplotlib.image as mpim

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

def generateHeatmap2(center_x, center_y, width, height):
    x = np.arange( width)
    y = np.arange( height)
    xv, yv = np.meshgrid(x, y)
    width_norm=0.01*np.sqrt(width*height)
    return np.exp(-0.5*((xv-center_x)**2+(yv-center_y)**2)/(width_norm**2))



class Dataset300WHM(Dataset):
    """Face Landmarks dataset."""

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
            pickle_path = os.path.join(self.input_path, f"pickle_300W_{self.max_size}_{self.num_landmarks}_col.p")
        else:
            pickle_path = os.path.join(self.input_path, f"pickle_300W_{self.max_size}_{self.num_landmarks}.p")
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
            if self.max_size==-1:
                number_of_images = len(image_paths)
            else:
                number_of_images = min(len(image_paths),self.max_size)
            with tqdm.tqdm(total=number_of_images , file=sys.stdout) as pbar_test:  # ini a progress bar
                for i, image_path in enumerate(image_paths):

                    pbar_test.set_description(
                        "Generate Images")  # update progress bar string output
                    pbar_test.update(1)  # update progress bar status

                    if (self.max_size != -1 and len(x) >= self.max_size):
                        break
                    # Resize the image to the input size
                    resized, points = resizeInput(image_path, landmarks[i], self.width_in, self.height_in)
                    x.append(resized)

                    # Scale the landmark coordinates to the output size
                    ratio = np.array([(self.width_in / self.width_out), (self.height_in / self.height_out)])
                    points = np.around(points / ratio, decimals=3)
                    # Get the heatmap for each landmark
                    if self.landmarks_collapsed:
                        u = np.zeros((self.width_out, self.height_out, 1))
                    else:
                        u = np.zeros((self.width_out, self.height_out, self.num_landmarks))
                    for j, (x_p, y_p) in enumerate(points[:self.num_landmarks]):
                        if self.landmarks_collapsed:
                            u[:, :, 0] +=generateHeatmap2(x_p, y_p, self.width_out, self.height_out)
                        else:
                            u[:, :, j] = generateHeatmap2(x_p, y_p, self.width_out, self.height_out)
                    u=np.clip(u,0,1)
                    y.append(u)
            self.x= np.transpose(np.array(x),(0,3,1,2))
            self.y= np.array(y)
            data =(self.x, self.y)
            pickle.dump(data, open(pickle_path, "wb"))
        self.length=len(self.x)

    def get_data(self, which_set):
        return self.x[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)], self.y[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)]


    def render(self, x,y,out,number_images):
        fig, ax = plt.subplots(nrows=number_images, ncols=3, figsize=(18, 3 * number_images))
        for row_num in range(number_images):
            x_img=np.transpose(x[row_num],(1,2,0))
            y_img = y[row_num][:,:,0]
            ax[row_num][0].imshow(cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB))
            ax[row_num][2].imshow(y_img)
            if type(out)!=type(None):
                out_img = out[row_num][:,:,0]
                ax[row_num][1].imshow(out_img)
        plt.show()
