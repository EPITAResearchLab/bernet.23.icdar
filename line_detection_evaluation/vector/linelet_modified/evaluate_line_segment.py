import cv2
import numpy as np
import os

from line_area_intersection import line_area_intersection
from compute_distance import are_angle_aligned, get_perp_dist, distance_L2
from utils import *

from typing import Tuple


def evaluate_line_segment(line_gnd : np.ndarray, line_est: np.ndarray, params: eval_param_struct) -> Tuple[float, float, float]:
    """
    Evaluate line segment detection
    :param line_gnd: Ground truth line segment
    :param line_est: Estimated line segment
    :param params: Parameters
    :return: Precision, recall, F-measure
    """
    # First line of gt contains x,y,height,width of pertinent area
    rect_data = line_gnd[0, :].astype(int)
    rect = rect_data[0, 0], rect_data[0, 1], rect_data[0, 2], rect_data[0, 3]
    # Remove first line
    line_gnd = line_gnd.copy()[1:, :]

    # Remove line with length < min_length using a for loop
    to_remove = []
    for i in range(line_gnd.shape[0]):
        if line_gnd[i, LENGTH_IDX] < params.min_length:
            to_remove.append(i)
    line_gnd = np.delete(line_gnd.copy(), to_remove, axis=0)

    # Clip line in pertinent area + remove line fully in non pertinent area
    to_remove = []
    for i in range(line_est.shape[0]):
        ret, (line_est[i, 0], line_est[i, 1]), (line_est[i, 2], line_est[i, 3]) = cv2.clipLine(
            rect, (int(line_est[i, 0]), int(line_est[i, 1])), (int(line_est[i, 2]), int(line_est[i, 3])))
        if not ret:
            to_remove.append(i)
    line_est = np.delete(line_est.copy(), to_remove, axis=0)

    if line_est.shape[0] == 0:
        if line_gnd.shape[0] == 0:
            return 1, 1, 1
        else:
            return 0, 0, 0
    if line_gnd.shape[0] == 0:
        return 0, 0, 0

    precision, recall = 0, 0

    # Initialize retrieval numbers -- 1st row: pixelwise, 2nd row: line segment wise
    tp_area_est, tp_area_gnd = 0, 0
    tp_inst_est, tp_inst_gnd = 0, 0
    fn_area_est, fn_area_gnd = 0, 0
    fn_inst_est, fn_inst_gnd = 0, 0

    tp_iou = 0
    fp_iou = 0
    fn_iou = 0

    num_gnd = line_gnd.shape[0]
    num_est = line_est.shape[0]

    # Involved one-to-N matching
    idx_match = np.full((num_est, num_gnd), False)

    for i_gnd in range(num_gnd):
        idx_perpd = np.transpose(get_perp_dist(line_gnd[i_gnd, CENTER_X_IDX:CENTER_Y_IDX + 1],
                                 line_est[:, CENTER_X_IDX: CENTER_Y_IDX + 1], line_gnd[i_gnd, ANGLE_IDX])) <= params.thres_dist
        idx_ang = are_angle_aligned(
            line_gnd[i_gnd, ANGLE_IDX], line_est[:, ANGLE_IDX], params.thres_ang)

        idx_cand = np.where(np.any((idx_perpd & idx_ang) == True, axis=1))[0]
        if idx_cand.shape[0] == 0:
            continue

        [gt_covered, idx_valid, _] = line_area_intersection(
            line_gnd[i_gnd, :], line_est[idx_cand, :])

        if idx_valid.shape[0] == 0 or (np.sum(gt_covered[idx_valid]) / line_gnd[i_gnd, LENGTH_IDX]) < params.thres_length_ratio:
            continue

        idx_match[idx_cand[idx_valid], i_gnd] = True

    est_with_multiple_match, *_ = np.where(idx_match.sum(axis=1) > 1)
    multi_match_est, multi_match_gnd = np.where(
        idx_match[est_with_multiple_match])

    k = 0
    for i_est in est_with_multiple_match:

        est_info = line_est[i_est, CENTER_X_IDX:CENTER_Y_IDX + 1]

        est_conflict = []
        while k < multi_match_est.shape[0] and est_with_multiple_match[multi_match_est[k]] == i_est:
            i_gnd = multi_match_gnd[k]

            gnd_info = line_gnd[i_gnd, CENTER_X_IDX:CENTER_Y_IDX + 1]

            dist_gnd_est = distance_L2(est_info, gnd_info)

            est_conflict.append((i_gnd, dist_gnd_est))
            k += 1

        est_conflict = sorted(est_conflict, key=lambda x: x[1])

        for i in range(1, len(est_conflict)):
            idx_match[i_est, est_conflict[i][0]] = False

    est_with_multiple_match, *_ = np.where(idx_match.sum(axis=1) > 1)
    assert est_with_multiple_match.shape[0] == 0

    for i_gnd in range(num_gnd):
        try:
            idx_cand = np.where(idx_match[:, i_gnd])[0]

            def false_negative():
                nonlocal fn_area_gnd, fn_inst_gnd, fn_iou

                fn_area_gnd += line_gnd[i_gnd, LENGTH_IDX]
                fn_inst_gnd += 1
                fn_iou += line_gnd[i_gnd, LENGTH_IDX]

            if idx_cand.shape[0] == 0:
                # False negative
                false_negative()
            else:
                # True positive
                [gt_covered, idx_valid, pd_covered] = line_area_intersection(
                    line_gnd[i_gnd, :], line_est[idx_cand, :])

                if idx_valid.shape[0] == 0 or (np.sum(gt_covered[idx_valid]) / line_gnd[i_gnd, LENGTH_IDX]) < params.thres_length_ratio:
                    # Should not happend : False True positive ???
                    false_negative()
                else:
                    if params.split_penalized:
                        # Punish over segmentation
                        tp_area_est += (np.sum(pd_covered[idx_valid]
                                               ) / idx_valid.shape[0])
                    else:
                        tp_area_est += np.sum(pd_covered[idx_valid])
                    tp_area_gnd += np.sum(gt_covered[idx_valid])
                    tp_inst_est += idx_valid.shape[0]  # Unused
                    tp_inst_gnd += 1  # Unused

                    tp_iou += np.sum(pd_covered[idx_valid])
                    fp_iou += np.sum(line_est[idx_cand[idx_valid],
                                     LENGTH_IDX]) - np.sum(pd_covered[idx_valid])
        except:
            # Give more information of the error
            print(f"error at evaluate_line_segment(), i_gnd: {i_gnd}.\n")

    precision_area_est = tp_area_est / np.sum(line_est[:, LENGTH_IDX])
    recall_area_gnd = tp_area_gnd / np.sum(line_gnd[:, LENGTH_IDX])

    precision = precision_area_est
    recall = recall_area_gnd
    iou = tp_iou / (tp_iou + fp_iou + fn_iou)

    return (precision, recall, iou)


def evaluate_line_segment_complete(g_csv : str, c_csv: str, params: eval_param_struct) -> Tuple[float, float, float, float]:
    """
    Evaluate the line segment detection result.
    :param g_csv: Ground truth csv file
    :param c_csv: Candidate csv file
    :param params: Evaluation parameters

    :return: Precision, Recall, IoU, F-score
    """
    if not os.path.exists(g_csv):
        return None, None, None, None

    if not os.path.exists(c_csv):
        return None, None, None, None

    line_ref = file_to_eval_line(g_csv)
    line_cand = file_to_eval_line(c_csv)

    (pr, re, iou) = evaluate_line_segment(line_ref, line_cand,
                                          params) if line_cand.shape[0] != 0 else (0, 0, 0)
    fsc = 0 if pr + re == 0 else 2 * pr * re / (pr + re)

    return (pr, re, iou, fsc)


if __name__ == '__main__':
    # ref_csv = # FIXME
    # cand_csv = # FIXME
    # params = evaluation_parameters = eval_param_struct(30, 3.14 * 5 / 180, .75, False, 300)
    # (pr, re, iou, fsc) = evaluate_line_segment_complete(ref_csv, cand_csv, params)
    # print(pr, re, iou, fsc)
    pass