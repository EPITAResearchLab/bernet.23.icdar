import numpy as np
from skimage.io import imread


def convert_bool(path: str) -> np.ndarray:
    """
    Convert image to boolean array
    :param path: path to image
    :return: boolean array
    """
    try:
        image = imread(path, as_gray=True)
    except:
        return None

    if image.dtype == 'float64':
        image = (image * 255).astype(int)

    image[image < 255] = 0

    arr = np.array(image, dtype=bool)
    arr = np.where(arr == True, False, True)

    return arr


def compute_score(ref: str, test: str):
    """
    Compute F1-score of
    - precision
    - recall
    - error rate

    :param ref: reference image
    :param test: test image
    :return: score
    """

    true_positive = np.count_nonzero(np.logical_and(ref, test) == True)
    false_positive = np.count_nonzero(np.logical_and(
        np.where(ref == True, False, True), test) == True)
    false_negative = np.count_nonzero(np.logical_and(
        ref, np.where(test == True, False, True)) == True)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1score = true_positive / \
        (true_positive + (false_negative + false_positive) / 2)

    return precision, recall, f1score


def icdar_2013_evaluate_pixel(ref_path: str, test_path: str):
    arr_ref = convert_bool(ref_path)
    arr_test = convert_bool(test_path)

    if arr_ref is None or arr_test is None:
        return -1

    # Compute error
    p, r, f = compute_score(arr_ref, arr_test)

    return p, r, f


if __name__ == "__main__":
    # name_ref = # FIXME
    # name_test = # FIXME
    # print(icdar_2013_evaluate_pixel(name_ref, name_test))

    # White => 255 => True
    # Black => 0   => False

    ref = np.array([[False, True, False, True, False]])

    test1 = np.array([[False, True, True, True, False]])
    print(compute_score(ref, test1))
    test2 = np.array([[False, True, False, True, False]])
    print(compute_score(ref, test2))
