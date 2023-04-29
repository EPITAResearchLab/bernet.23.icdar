import numpy as np
from math import pi

X1_IDX = 0
Y1_IDX = 1
X2_IDX = 2
Y2_IDX = 3

CENTER_X_IDX = 4
CENTER_Y_IDX = 5

LENGTH_IDX = 6
ANGLE_IDX = 7


def read_csv(filename):
    with open(filename) as file_name:
        ret = np.loadtxt(file_name, delimiter=",")
    return ret


def line_to_eval_line(lines):
    # Compute x and y centers
    cp = np.matrix((lines[:, X1_IDX] + lines[:, X2_IDX],
                   lines[:, Y1_IDX] + lines[:, Y2_IDX])).T / 2
    dx = lines[:, X2_IDX] - lines[:, X1_IDX]  # Compute x distance
    dy = lines[:, Y2_IDX] - lines[:, Y1_IDX]  # Compute y distance
    length = np.matrix(np.sqrt(np.square(dx) + np.square(dy))
                       ).T  # Length of each line
    ang = np.matrix(np.arctan2(dy, dx)).T  # Angle of each line

    ret = np.concatenate((lines, cp, length, ang), axis=1)
    return ret


def file_to_eval_line(filename):
    lines_raw = read_csv(filename)
    if lines_raw.shape == (0, ):
        return lines_raw
    ret = line_to_eval_line(lines_raw)
    return ret


class eval_param_struct:
    def __init__(self, thres_dist, thres_ang, thres_length_ratio) -> None:
        self.thres_dist = thres_dist
        self.thres_ang = thres_ang
        self.thres_length_ratio = thres_length_ratio
