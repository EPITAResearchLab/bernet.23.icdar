import numpy as np
from math import pi, sin, cos, sqrt

from typing import Tuple

def get_perp_dist(pt0: np.ndarray, ptSet: np.ndarray, angle: float):
    """
    This function compute the perpendicular distance between a vector and a set of points
    :param pt0 : Grount truth center X and center Y of the current vector
    :param ptSet : Center X and center Y of all observations
    :param angle : Angle of the current ground truth vector

    :return A numpy array containing the perpendiculare distance for each observation to the current ground truth tested
    """
    dirVec = np.matrix([-sin(angle), cos(angle)])
    perpDist = np.absolute(np.matmul(dirVec, np.transpose(
        ptSet - np.tile(pt0, (ptSet.shape[0], 1)))))
    return perpDist


def are_angle_aligned(gtAng: np.ndarray, ang2: np.ndarray, thres: float) -> np.ndarray:
    """
    This function check if the angle of the ground truth vector and the angle of the observation vector are aligned

    :param gtAng : Angle of the ground truth vector
    :param ang2 : Angle of the observation vector
    :param thres : Threshold to determine if the angle are aligned

    :return A numpy array containing True if the angle are aligned and False otherwise
    """
    # Compute positive Ang for GT
    if gtAng < 0:
        gtAng += pi
    gtAng %= pi

    # Compute positive Ang for observations
    ang2 = np.where(ang2 < 0, ang2 + pi, ang2)
    ang2 %= pi

    ang = ang2 - gtAng
    ang = np.where(ang < 0, -ang, ang)
    ang = np.where(ang > pi * 3 / 4, pi - ang, ang)

    ret = ang <= thres
    return ret


def distance_L2(p1 : Tuple[int, int], p2: Tuple[int, int]):
    """
    Compute the L2 distance between two points

    :param p1 : First point
    :param p2 : Second point

    :return The L2 distance between the two points
    """
    x1, y1 = p1[0, 0], p1[0, 1]
    x2, y2 = p2[0, 0], p2[0, 1]

    d = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return d
