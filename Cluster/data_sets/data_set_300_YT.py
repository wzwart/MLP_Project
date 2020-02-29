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



def generateHeatmap(center_x, center_y, width, height,rbf_width):
    x = np.arange( width)
    y = np.arange( height)
    xv, yv = np.meshgrid(x, y)
    width_norm=rbf_width  *np.sqrt(width*height)
    hm= np.exp(-0.5*((xv-center_x)**2+(yv-center_y)**2)/(width_norm**2))
    # hm = hm - hm.mean(axis=(0, 1))
    # but don't normalize variance
    return hm


class Dataset_300W_YT(Dataset):
    """300W and Youtube Faces datasets"""

    def __init__(self, root_dir, width_in,height_in, width_out, height_out, num_landmarks,rbf_width, which_dataset, landmarks_collapsed=False, max_size= -1):
        """
        Args:
            root_dir: Data directory containing both 300W and Youtube Faces directories.
            width_in: Width of the input image.
            height_in: Height of the input image.
            width_out: Width of the output heatmap.
            height_out: Height of the output heatmap.
            num_landmarks: Number of landmarks to do the heatmap for.
            which_dataset: Use 300W (0), Youtube faces (1) or both (2)
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
        self.landmarks_collapsed=landmarks_collapsed
        self.frac = {"train": (0, 0.7), "valid": (0.7, 0.9), "test": (0.9, 1)}
        self.create_dataset()

    def __len__(self):
        return self.length

    def create_dataset(self):
        import pickle
        if self.landmarks_collapsed:
            if(self.which_dataset == 0):
                pickle_path = os.path.join(self.input_path, f"pickle_300W_{self.max_size}_{self.num_landmarks}_col.p")
            elif(self.which_dataset == 1):
                pickle_path = os.path.join(self.input_path, f"pickle_Youtube_{self.max_size}_{self.num_landmarks}_col.p")
            elif (self.which_dataset == 2):
                pickle_path = os.path.join(self.input_path, f"pickle_300W_Youtube_{self.max_size}_{self.num_landmarks}_col.p")
            else:
                raise ValueError

        else:
            if (self.which_dataset == 0):
                pickle_path = os.path.join(self.input_path, f"pickle_300W_{self.max_size}_{self.num_landmarks}.p")
            elif (self.which_dataset == 1):
                pickle_path = os.path.join(self.input_path, f"pickle_Youtube_{self.max_size}_{self.num_landmarks}.p")
            elif (self.which_dataset == 2):
                pickle_path = os.path.join(self.input_path, f"pickle_300W_Youtube_{self.max_size}_{self.num_landmarks}.p")
            else:
                raise ValueError

        if os.path.exists(pickle_path):
            print("loading from pickle file")
            data = pickle.load(open(pickle_path, "rb"))
            (self.x, self.y, self.p) = data
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

            x = []
            y = []
            p = []
            if self.max_size == -1:
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

                    # normalize:
                    resized = resized-resized.mean(axis=(0,1))
                    resized = resized / np.sqrt(resized.var(axis=(0,1)))

                    x.append(resized)

                    # Scale the landmark coordinates to the output size
                    ratio = np.array([(self.width_in / self.width_out), (self.height_in / self.height_out)])
                    points = np.around(points / ratio)
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
                    y.append(u)
                    p.append(points[:self.num_landmarks])

            self.x= np.transpose(np.array(x),(0,3,1,2))
            self.y= np.array(y)
            self.p = np.array(p)

            self.x, self.y, self.p = shuffle(self.x, self.y, self.p, random_state=0)
            data =(self.x, self.y, self.p)
            pickle.dump(data, open(pickle_path, "wb"))
        self.length=len(self.x)

    def get_data(self, which_set):
        return self.x[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)], self.y[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)], self.p[int(self.frac[which_set][0]*self.length):int(self.frac[which_set][1]*self.length)]

    def NME(self, true, predicted):

        bbox = [128,128]
        #bbox = np.max(true, axis=0) - np.min(true, axis=0)
        print(bbox)
        print(true)
        d = np.sqrt(bbox[0]*bbox[1])
        twonorm = np.sqrt(np.sum(np.square(true-predicted)))
        return (twonorm/d)/len(true)


    def render(self, x,y,p,out,number_images):
        from collections import OrderedDict
        from matplotlib import cm

        set1 = cm.get_cmap('Set1')
        colors= np.asarray(set1.colors)
        no_colors=colors.shape[0]
        no_landmarks=y.shape[3]
        if type(out) != type(None):
            no_cols = 2+no_landmarks
        else:
            no_cols = 2
        fig, ax = plt.subplots(nrows=number_images, ncols=no_cols, figsize=(18, 3 * number_images))
        nme = 0
        count = 0

        u = np.zeros((self.width_out, self.height_out, 3))
        u[:, :, 0] = generateHeatmap(int(self.width_out / 2), int(self.height_out / 2), self.width_out, self.height_out, self.rbf_width)
        u[:, :, 1] = generateHeatmap(int(self.width_out / 2), int(self.height_out / 2), self.width_out, self.height_out, self.rbf_width)
        u[:, :, 2] = generateHeatmap(int(self.width_out / 2), int(self.height_out / 2), self.width_out, self.height_out, self.rbf_width)

        u = np.clip(u, 0, 1)

        u = np.array([u.transpose((2, 0, 1))])
        for row_num in range(number_images):
            x_img=np.transpose(x[row_num],(1,2,0))

            x_img=x_img-np.min(x_img, axis=(0,1))
            x_img=x_img/np.max(x_img, axis=(0,1))

            y_img = np.array([np.array([y[row_num,:,:,i]*colors[i%no_colors,0], y[row_num,:,:,i]*colors[i%no_colors,1],y[row_num,:,:,i]*colors[i%no_colors,2]]) for i in range(no_landmarks)])
            y_img = np.sum(y_img,axis=0).transpose((1,2,0))

            y_img=y_img-np.min(y_img, axis=(0,1))
            # y_img=y_img/np.max(y_img, axis=(0,1))

            p_img=p[row_num]

            x_img = x_img[:, :, [2, 1, 0]]# RGB BGR conversion

            bw_image = 0.3 * np.array([np.mean(x_img, axis=2), np.mean(x_img, axis=2), np.mean(x_img, axis=2)]).transpose((1, 2, 0))
            ax[row_num][0].imshow(x_img)
            ax[row_num][0].axis('off')
            ax[row_num][no_cols-1].imshow(y_img+bw_image)
            ax[row_num][no_cols-1].axis('off')

            #print("OUT {}".format(out.shape))
            #print(x_img.shape)
            #print(y_img.shape)
            #print(torch.Tensor(u).float().shape)
            #print(np.array([np.mean(x_img, axis=2)])[0].shape)
            #print(bw_image.shape)
            if type(out)!=type(None):
                predicted = np.array([])
                for i in range(no_landmarks):
                    out_img= (out[row_num] - out[row_num].min())
                    #print("OUTIMG {}".format(out_img[:,:,i].shape))
                    out_img = np.array([out_img[:,:,i]*colors[i%no_colors,0], out_img[:,:,i]*colors[i%no_colors,1],out_img[:,:,i]*colors[i%no_colors,2]]).transpose((1,2,0))
                    #print("OUTIMG {}".format(out_img.shape))
                    out_conv = np.array([out_img.transpose((2, 0, 1))])
                    padded_tensor = F.pad(torch.Tensor(out_conv).float(), (
                        int(self.height_out / 2), int(self.height_out / 2) - 1, int(self.width_out / 2),
                        int(self.width_out / 2) - 1))
                    cross_corr = F.conv2d(padded_tensor, torch.Tensor(u).float(), padding=0)

                    index_i, index_j = np.unravel_index(cross_corr.numpy()[0][0].argmax(), cross_corr.numpy()[0][0].shape)
                    print(index_j, index_i)
                    print("p_img: {}".format(p_img[i]))
                    if(i==0):
                        predicted = np.hstack((predicted,np.array([index_j, index_i])))
                    else:
                        predicted = np.vstack((predicted, np.array([index_j, index_i])))
                    #print(index_j, index_i)
                    #print(p_img[i])
                    cv2.circle(cross_corr.numpy()[0][0], (int(index_j), int(index_i)), 2, (0, 255, 0), -1)
                    cv2.circle(cross_corr.numpy()[0][0], (int(p_img[i][0]), int(p_img[i][1])), 2, (0, 0, 255), -1)
                    ax[row_num][i+1].imshow(cross_corr.numpy()[0][0]+np.array([np.mean(x_img, axis=2)])[0])
                    ax[row_num][i+1].axis('off')

                nme += self.NME(p_img,predicted)
                print(predicted)
                count += 1
            #plt.tight_layout()
            '''
            out_img = (out[row_num] - out[row_num].min())
            out_img = np.array(
                [out_img[:, :, 0] * colors[0 % no_colors, 0], out_img[:, :, 0] * colors[0 % no_colors, 1],
                 out_img[:, :, 0] * colors[0 % no_colors, 2]]).transpose((1, 2, 0))
            out_conv = np.array([out_img.transpose((2, 0, 1))])
            padded_tensor = F.pad(torch.Tensor(out_conv).float(), (
            int(self.height_out / 2), int(self.height_out / 2) - 1, int(self.width_out / 2),
            int(self.width_out / 2) - 1))
            cross_corr = F.conv2d(padded_tensor, torch.Tensor(u).float(), padding=0)
            print(cross_corr.shape)
            i, j = np.unravel_index(cross_corr.numpy()[0][0].argmax(), cross_corr.numpy()[0][0].shape)
            print(i,j)
            cv2.circle(cross_corr.numpy()[0][0], (int(j), int(i)), 2, (255, 0, 0), -1)
            ax[row_num][no_landmarks + 2].imshow(cross_corr.numpy()[0][0])
            ax[row_num][no_landmarks + 2].axis('off')
            '''
        print("NME: {}".format(nme/count))
        plt.show()

        '''
        out = torch.Tensor(out).float()
        set1 = cm.get_cmap('Set1')
        colors = np.asarray(set1.colors)
        no_colors = colors.shape[0]
        no_landmarks = out.shape[3]

        nme = 0
        count = 0
        height_out = out.shape[1]
        width_out = out.shape[2]
        n_images = out.shape[0]
        u = np.zeros((width_out, height_out, 3))
        u[:, :, 0] = generateHeatmap(int(width_out / 2), int(height_out / 2), width_out, height_out)
        u[:, :, 1] = generateHeatmap(int(width_out / 2), int(height_out / 2), width_out, height_out)
        u[:, :, 2] = generateHeatmap(int(width_out / 2), int(height_out / 2), width_out, height_out)

        u = np.clip(u, 0, 1)

        u = np.array([u.transpose((2, 0, 1))])
        u = torch.Tensor(u).float().to(torch.cuda.current_device())
        for row_num in range(n_images):
            if type(out) != type(None):
                predicted = np.array([])
                p_img = p[row_num]
                for i in range(no_landmarks):
                    out_img = (out[row_num] - out[row_num].min())
                    # print(out_img.shape)
                    t = torch.Tensor(3, out_img.shape[0], out_img.shape[1])
                    t[0] = out_img[:, :, i] * colors[i % no_colors, 0]
                    t[1] = out_img[:, :, i] * colors[i % no_colors, 1]
                    t[2] = out_img[:, :, i] * colors[i % no_colors, 2]
                    t = t.permute((1, 2, 0))

                    out_conv = t.permute((2, 0, 1)).unsqueeze(0)
                    out_conv = out_conv.to(torch.cuda.current_device())
                    padded_tensor = F.pad(out_conv, (
                        int(height_out / 2), int(height_out / 2) - 1, int(width_out / 2),
                        int(width_out / 2) - 1))
                    cross_corr = F.conv2d(padded_tensor, u, padding=0)

                    argmax = torch.argmax(cross_corr)

                    index_i, index_j = np.unravel_index(argmax.detach().cpu(),
                                                        (out_img.shape[0], out_img.shape[1]))
                    print(index_j, index_i)
                    if (i == 0):
                        predicted = np.hstack((predicted, np.array([index_j, index_i])))
                    else:
                        predicted = np.vstack((predicted, np.array([index_j, index_i])))

                nme += self.NME(p_img, predicted)
                count += 1

        print("NME: {}".format(nme/count))
        '''
