import numpy as np
import cv2


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
    return hm
