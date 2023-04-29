import numpy as np

X1_IDX = 0
Y1_IDX = 1
X2_IDX = 2
Y2_IDX = 3

CENTER_X_IDX = 4
CENTER_Y_IDX = 5

LENGTH_IDX = 6
ANGLE_IDX = 7


def read_csv(filename: str) -> np.ndarray:
    """
    Read a csv file
    :param filename: The path of the csv file
    :return: A numpy array containing the data of the csv file
    """
    with open(filename, encoding="utf8", errors='ignore') as file_name:
        ret = np.loadtxt(file_name, delimiter=",")
    if ret.shape == (4, ):
        ret = ret.reshape((1, 4))
    return ret


def line_to_eval_line(lines: np.ndarray) -> np.ndarray:
    """
    Convert a line to a line with center point, length and angle
    :param lines: A numpy array containing the lines
    :return: A numpy array containing the lines with center point, length and angle
    """
    cp = np.matrix((lines[:, X1_IDX] + lines[:, X2_IDX],
                   lines[:, Y1_IDX] + lines[:, Y2_IDX])).T / 2
    dx = lines[:, X2_IDX] - lines[:, X1_IDX]  # Compute x distance
    dy = lines[:, Y2_IDX] - lines[:, Y1_IDX]  # Compute y distance
    length = np.matrix(np.sqrt(np.square(dx) + np.square(dy))
                       ).T  # Length of each line
    ang = np.matrix(np.arctan2(dy, dx)).T  # Angle of each line
    ret = np.concatenate((lines, cp, length, ang), axis=1)
    return ret


def file_to_eval_line(filename: str) -> np.ndarray:
    """
    Read a csv file and convert it to a line with center point, length and angle
    :param filename: The path of the csv file
    :return: A numpy array containing the lines with center point, length and angle
    """
    lines_raw = read_csv(filename)
    if lines_raw.shape == (0, ):
        return lines_raw
    if lines_raw.shape == (4, ):
        lines_raw = lines_raw.reshape((1, 4))
    lines = line_to_eval_line(lines_raw)
    return lines


class eval_param_struct:
    """
    The parameters for the evaluation

    :param thres_dist: The threshold for the perpendicular distance
    :param thres_ang: The threshold for the angle in radians
    :param thres_length_ratio: The threshold for the length ratio
    :param split_penalized: If the split is penalized
    :param min_length: The minimum length of a line
    """

    def __init__(self, thres_dist: float, thres_ang: float, thres_length_ratio: float, split_penalized: float = False, min_length: float = 0) -> None:
        self.thres_dist = thres_dist
        self.thres_ang = thres_ang
        self.thres_length_ratio = thres_length_ratio
        self.split_penalized = split_penalized
        self.min_length = min_length
