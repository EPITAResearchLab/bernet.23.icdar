
from coco_pano_ext_demo import COCO, COCO_plot, precision_recall_maps

import numpy as np
from skimage.io import imread
from skimage.color import gray2rgb


white = 255 * 256 ** 2 + 255 * 256 ** 1 + 255
black = 0 * 256 ** 2 + 0 * 256 ** 1 + 0


def rgb2label(rgb_img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to label image
    :param rgb_img: RGB image
    :return: label image
    """
    rgb_img_2d = rgb_img[..., 0] * \
        (256 ** 2) + rgb_img[..., 1] * 256 + rgb_img[..., 2] * 1
    index, img_label = np.unique(rgb_img_2d, return_inverse=True)

    background_color = black
    try:
        index_background = list(index).index(background_color)
    except:
        background_color = white
        index_background = list(index).index(background_color)

    # Set background to 0
    img_label = np.where(img_label == index_background, -1, img_label)
    img_label = np.where(img_label == 0, index_background, img_label)
    img_label = np.where(img_label == -1, 0, img_label)

    img_label = img_label.reshape(rgb_img.shape[:2])

    return img_label


def imreadlabel(path: str, inverted: bool = False):
    """
    Read image and convert to label image
    :param path: path to image
    :param inverted: invert image
    :return: label image
    """
    img = imread(path, as_gray=False)
    if len(img.shape) == 2:
        img = gray2rgb(img)

    label_img = rgb2label(img)

    return label_img


def eval_coco_path(gt_path: str, pr_path: str):
    """
    Evaluate COCO score and precision-recall map
    :param gt_path: path to ground truth image
    :param pr_path: path to prediction image
    :return: COCO score and precision-recall map
    """
    T = imreadlabel(gt_path)
    P = imreadlabel(pr_path)

    coco_score = COCO(P, T, ignore_zero=True, output_scores=True)
    coco_img = precision_recall_maps(T, P, lower_bound=0.5)

    return *coco_score, *coco_img


def eval_coco_path_score(gt_path: str, pr_path: str):
    """
    Evaluate COCO score
    :param gt_path: path to ground truth image
    :param pr_path: path to prediction image
    :return: COCO score
    """

    T = imreadlabel(gt_path)
    P = imreadlabel(pr_path)
    coco_score = COCO(P, T, ignore_zero=True, output_scores=True)
    return coco_score[:3]
