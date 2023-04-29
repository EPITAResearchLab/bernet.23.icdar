import numpy as np
from math import pi, sin, cos


def get_perp_dist(pt0, ptSet, angle):
    """
    # FIXME : Documentation

    pt0 : Grount truth center X and center Y of the current vector
    ptSet : Center X and center Y of all observations
    angle : Angle of the current ground truth vector

    return : A numpy array containing the perpendiculare distance for each observation to the current ground truth tested
    """
    dirVec = np.matrix([-sin(angle), cos(angle)])
    perpDist = np.absolute(np.matmul(dirVec, np.transpose(
        ptSet - np.tile(pt0, (ptSet.shape[0], 1)))))
    return perpDist


def are_angle_aligned(gtAng, ang2, thres):
    """
    # FIXME : Documentation
    gtAng : Ground truth angle
    ang2 : Matrix of angle of observations
    thres : Threshold of angle difference

    return : A numpy array of bool saiying if observation i is aligned to gt vector
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
