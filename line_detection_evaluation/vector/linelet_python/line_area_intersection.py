import numpy as np
from utils import *


def line_area_intersection(l_src, l_tar):
    """
    FIXME : Documentation
    l_src : One line containing current ground truth vector data
    l_tar : Containes one or multiple vector matching l_src vector
    """
    nb_line_tar = l_tar.shape[0]
    idx_valid = np.full((nb_line_tar, 1), False)

    gt_covered = np.zeros((nb_line_tar, 1))
    pd_covered = np.zeros((nb_line_tar, 1))

    # Project the source to the source coordinate
    vec_base = l_src[:, [X2_IDX, Y2_IDX]] - l_src[:, [X1_IDX, Y1_IDX]]
    vec_base = np.transpose(vec_base / np.linalg.norm(vec_base))

    m = np.concatenate((l_src[:, [X1_IDX, Y1_IDX]] - l_src[:, [X1_IDX, Y1_IDX]],
                       l_src[:, [X2_IDX, Y2_IDX]] - l_src[:, [X1_IDX, Y1_IDX]]))
    vec_src = np.transpose(np.matmul(m, vec_base))

    if vec_src[0, 0] > vec_src[0, 1]:
        vec_src[0, 1], vec_src[0, 0] = vec_src[0, 0], vec_src[0, 1]

    # FIXME
    l1 = np.matmul(l_tar[:, [X1_IDX, Y1_IDX]] - np.tile(l_src[:,
                   [X1_IDX, Y1_IDX]], (nb_line_tar, 1)), vec_base)
    l2 = np.matmul(l_tar[:, [X2_IDX, Y2_IDX]] - np.tile(l_src[:,
                   [X1_IDX, Y1_IDX]], (nb_line_tar, 1)), vec_base)
    vec_tar = np.concatenate((l1, l2), axis=1)

    for i in range(nb_line_tar):
        if vec_tar[i, 0] > vec_tar[i, 1]:
            vec_tar[i, 0], vec_tar[i, 1] = vec_tar[i, 1], vec_tar[i, 0]

    # clip left area
    vec_tar[vec_tar < 0] = 0

    #  clip right area
    vec_tar[vec_tar > np.max(vec_src)] = np.max(vec_src)

    for k in range(nb_line_tar):
        # Project the target to the source's coordinate
        bValid = True

        si, sf = vec_src[0, 0], vec_src[0, 1]
        # src : si------sf
        ti, tf = vec_tar[k, 0], vec_tar[k, 1]
        # tar : ti------tf

        if ti >= sf or tf <= si:
            # case 1
            # tar:                  *----------*
            # src: *---------*
            # case 2
            # tar: *----------*
            # src:                *---------*
            gt_covered[k] = 0
            idx_valid[k] = False
            pd_covered[k] = 0
            bValid = False
        elif ti <= si and tf >= si and tf <= sf:
            # case 3
            # tar: *-------*
            # src:    *---------*
            vec_tar[k, 0] = si
            # tar:    *----*
        elif tf >= sf and ti >= si and ti <= sf:
            # case 4
            # tar:      *----------*
            # src: *---------*
            vec_tar[k, 1] = sf
            # tar:      *----*
        elif ti <= si and sf <= tf:
            # case 5
            # tar: *-------------------*
            # src:     *---------*
            vec_tar[k, :] = vec_src
            # tar:     *---------*

        if bValid:
            idx_valid[k] = True
            pd_covered[k] = np.absolute(
                vec_tar[k, 0] - vec_tar[k, 1])  # area covered by target

    for i1 in range(nb_line_tar - 1):
        if not idx_valid[i1]:
            continue

        for i2 in range(i1 + 1, nb_line_tar):
            if not idx_valid[i2]:
                continue

            ts1, tf1 = vec_tar[i1, 0], vec_tar[i1, 1]
            ts2, tf2 = vec_tar[i2, 0], vec_tar[i2, 1]

            if ts2 <= ts1 and tf2 <= tf1 and tf2 >= ts1:
                # tar1:    *--------*
                # tar2: *-------*
                vec_tar[i2, 1] = ts1
                # tar2: *--*

            if ts2 >= ts1 and tf2 >= tf1 and ts2 <= tf1:
                # tar1: *----------*
                # tar2:     *--------*
                vec_tar[i2, 0] = tf1
                # tar2:            *-*

            if ts1 <= ts2 and tf2 <= tf1:
                # tar1: *----------*
                # tar2:     *----*
                vec_tar[i2, :] = 0
                # tar2:
                
            elif ts2 <= ts1 and tf1 <= tf2:
                # tar1:     *----*
                # tar2: *----------*
                vec_tar[i1, :] = 0
                # tar1:

    for k in range(nb_line_tar):
        gt_covered[k] = vec_tar[k, 1] - vec_tar[k, 0]

    idx_valid = np.where(np.any(idx_valid == True, axis=1))[0]

    return gt_covered, idx_valid, pd_covered
