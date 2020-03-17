import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def render(x, y, p, n, out, nme_results, number_images, experiment):
    set1 = cm.get_cmap('Set1')
    colors = np.asarray(set1.colors)
    no_colors = colors.shape[0]
    no_landmarks = y.shape[3]
    no_cols = 8
    five_land = [33, 36, 39, 42, 45]
    fig, ax = plt.subplots(nrows=number_images, ncols=no_cols, figsize=(18, 3 * number_images))
    if number_images == 1:
        ax = [ax]
    nme, sqrt_errors, norm_array = nme_results

    for row_num in range(number_images):
        x_img = np.transpose(x[row_num], (1, 2, 0))
        x_img = x_img - np.min(x_img, axis=(0, 1))
        x_img = x_img / np.max(x_img, axis=(0, 1))
        y_img = np.array([np.array(
            [y[row_num, :, :, i] * colors[i % no_colors, 0], y[row_num, :, :, i] * colors[i % no_colors, 1],
             y[row_num, :, :, i] * colors[i % no_colors, 2]]) for i in range(no_landmarks)])
        y_img = np.sum(y_img, axis=0).transpose((1, 2, 0))
        y_img = y_img - np.min(y_img, axis=(0, 1))
        x_img = x_img[:, :, [2, 1, 0]]  # RGB BGR conversion
        bw_image = 0.4 * np.array(
            [np.mean(x_img, axis=2), np.mean(x_img, axis=2), np.mean(x_img, axis=2)]).transpose((1, 2, 0))
        ax[row_num][0].imshow(x_img)
        ax[row_num][0].axis('off')
        ax[row_num][no_cols - 1].imshow(np.clip(y_img + bw_image, 0, 1))
        ax[row_num][no_cols - 1].axis('off')
        five_counter = 1
        if type(out) != type(None):
            sum_img = np.zeros((out.shape[1], out.shape[2], 3))
            for i in range(no_landmarks):
                out_img = out[row_num, :, :, i]
                kth = int(.9 * out_img.size)
                out_img_min = np.max(np.partition(out_img.flatten(), kth)[:kth])
                out_img = np.clip(out_img - out_img_min, 0, None)

                if np.max(out_img) != 0:
                    out_img = out_img / (np.max(out_img))
                out_img = np.array(
                    [out_img * colors[i % no_colors, 0], out_img * colors[i % no_colors, 1],
                     out_img * colors[i % no_colors, 2]]).transpose(1, 2, 0)
                if (i in five_land):
                    ax[row_num][five_counter].imshow(np.clip(out_img + bw_image, 0, 1))
                    ax[row_num][five_counter].axis('off')
                    five_counter += 1
                    ax[row_num][five_counter].text(x=0.0, y=-0.1,
                                                   s=f"NE= {100 * sqrt_errors[row_num][i] / norm_array[row_num]:.1f}%",
                                                   horizontalalignment='left', verticalalignment='top',
                                                   transform=ax[row_num][five_counter - 1].transAxes)
                sum_img = sum_img + out_img

            ax[row_num][6].imshow(np.clip(sum_img + bw_image, 0, 1))
            # ax[row_num][6].imshow(np.clip(sum_img,0,1))
            normalised_errors = (sqrt_errors.T / norm_array).T
            ax[row_num][6].text(x=0.0, y=-0.1, s=f"NME= {100 * np.mean(normalised_errors[row_num]):.1f}%",
                                horizontalalignment='left', verticalalignment='top', transform=ax[row_num][6].transAxes)

            ax[row_num][6].axis('off')

    directory = experiment + '/render.pdf'
    plt.savefig(directory)
    plt.show()
