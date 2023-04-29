import numpy as np
from skimage.io import imread


def convert_bool(path) -> np.ndarray:
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


def compute_score(ref, test):
    """
    Compute ICDAR 2011 evaluation score

    :param ref: reference image
    :param test: test image
    :return: score
    """
    # Compute values
    nb_pixel = 1
    for s in ref.shape:
        nb_pixel *= s

    # arr_xor : false positive + false negative
    arr_xor = np.logical_xor(ref, test)

    bad_val = np.count_nonzero(arr_xor == True)

    # Error
    e = bad_val / nb_pixel

    return 1 - e


def icdar_2011_evaluate_pixel(ref_path : str, test_path : str):
    """
    Evaluate pixel of ICDAR 2011

    :param ref_path: path to reference image
    :param test_path: path to test image
    :return: score
    """
    # Work on images
    arr_ref = convert_bool(ref_path)
    arr_test = convert_bool(test_path)

    if arr_ref is None or arr_test is None:
        return -1

    # Compute error
    s = compute_score(arr_ref, arr_test)

    return s


if __name__ == "__main__":
    # name_ref = # FIXME
    # name_test = # FIXME
    # print(evaluate_pixel(name_ref, name_test))

    # White => 255 => True
    # Black => 0   => False

    ref = np.array([[False, True, False, True, False]])

    test1 = np.array([[False, True, True, True, False]])
    print(compute_score(ref, test1))
    test2 = np.array([[False, True, False, True, False]])
    print(compute_score(ref, test2))
