import cv2
import numpy as np
import os

from line_area_intersection import line_area_intersection
from compute_distance import are_angle_aligned, get_perp_dist
from utils import *


def evaluate_line_segment(line_gnd, line_est, params):
    """
        # FIXME : Documentation
        Evaluate line segmentation

        line_gnd : Line segment instance of the ground truth
        line_gnd : Line segment instance of a technique
        params : Parameters used in the evaluation

        Line segment instance should be in a form (x1, y1, x2, y2, center_x, center_y, length, angle)
    """
    # First line of gt contains x,y,height,width of pertinent area
    line_gnd = line_gnd.copy()[1:, :]
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

    for i_gnd in range(num_gnd):
        try:
            # line structure: (x1, y1, x2, y2, center_x, center_y, length, angle)
            idx_perpd = np.transpose(get_perp_dist(line_gnd[i_gnd, CENTER_X_IDX:CENTER_Y_IDX + 1],
                                     line_est[:, CENTER_X_IDX: CENTER_Y_IDX + 1], line_gnd[i_gnd, ANGLE_IDX])) <= params.thres_dist
            idx_ang = are_angle_aligned(
                line_gnd[i_gnd, ANGLE_IDX], line_est[:, ANGLE_IDX], params.thres_ang)
            idx_cand = np.where(
                np.any((idx_perpd & idx_ang) == True, axis=1))[0]

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


def evaluate_line_segment_complete(g_csv, c_csv, params):
    if not os.path.exists(g_csv) or not os.path.exists(c_csv):
        return None, None, None, None

    line_ref = file_to_eval_line(g_csv)
    line_cand = file_to_eval_line(c_csv)

    (pr, re, iou) = evaluate_line_segment(line_ref, line_cand,
                                          params) if line_cand.shape[0] != 0 else (0, 0, 0)
    fsc = 0 if pr + re == 0 else 2 * pr * re / (pr + re)

    return (pr, re, iou, fsc)


if __name__ == '__main__':
    curr_dir = "../../../"
    ref_csv = curr_dir + "data/csv/source/annuaire/2.csv"
    cand_csv = curr_dir + "out.csv"
    params = eval_param_struct(50, pi * 5 / 180, .75)

    (pr, re, iou, fsc) = evaluate_line_segment_complete(ref_csv, ref_csv, params)

    print(pr, re, iou, fsc)
