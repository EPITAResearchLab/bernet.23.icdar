import csv
import numpy as np
import os

from math import pi
from matplotlib import pyplot as plt
from statistics import mean

from evaluate_line_segment import evaluate_line_segment
from utils import file_to_eval_line, eval_param_struct


class stats_method_struct:
    def __init__(self) -> None:
        self.prec = []
        self.rec = []
        self.iou = []


if __name__ == '__main__':
    dir_db = './data/csv'

    with open('./evaluation/Image_ID_List.csv') as file:
        csvreader = csv.reader(file)
        Image_ID_List = [row[0] for row in csvreader]
    num_im = len(Image_ID_List)

    dir_method = dir_db + "/lsd/"
    method2test = ['von', 'OCV_Hough', 'edlines',
                   'cannylines', 'elsed', 'pylene']
    num_method = len(method2test)

    stats = [stats_method_struct() for _ in range(num_method)]

    # True positive conditions
    eval_param = eval_param_struct(2, pi * 5 / 180, .75)

    for i_im in range(num_im):
        try:
            #  load line gnd
            str_gnd = dir_db + "/source/" + Image_ID_List[i_im] + ".csv"

            if not os.path.exists(str_gnd):
                continue

            lines_gnd = file_to_eval_line(str_gnd)

            for k in range(num_method):
                #  Load estimation results
                str_est = dir_db + "/lsd/" + \
                    method2test[k] + "/" + Image_ID_List[i_im] + ".csv"
                if not os.path.exists(str_est):
                    continue

                lines_est = file_to_eval_line(str_est)

                #  Evaluate
                if lines_est.shape[0] > 0:
                    (pr, re, iou) = evaluate_line_segment(
                        lines_gnd, lines_est, eval_param)

                    stats[k].prec.append(pr)
                    stats[k].rec.append(re)
                    stats[k].iou.append(iou)
                else:
                    stats[k].prec.append(0)
                    stats[k].rec.append(0)
                    stats[k].iou.append(0)

        except:
            print(f"Error at i_im: {i_im}.\n")

    # Display scores
    AP = np.zeros((num_method, 1))
    AR = np.zeros((num_method, 1))
    IOU = np.zeros((num_method, 1))

    for k in range(num_method):
        if len(stats[k].prec) == 0:
            continue
        AP[k, 0] = mean(stats[k].prec)
        AR[k, 0] = mean(stats[k].rec)
        IOU[k, 0] = mean(stats[k].iou)

    F_sc = 2 * (np.multiply(AP, AR)) / (AP + AR)

    AP = AP.T.tolist()[0]
    AR = AR.T.tolist()[0]
    IOU = IOU.T.tolist()[0]
    F_sc = F_sc.T.tolist()[0]

    X_axis = np.arange(num_method)

    plt.bar(X_axis - 0.3, AP, 0.2, label='AP')
    plt.bar(X_axis - 0.1, AR, 0.2, label='AR')
    plt.bar(X_axis + 0.1, IOU, 0.2, label='IOU')
    plt.bar(X_axis + 0.3, F_sc, 0.2, label='F_Sc')

    plt.xticks(X_axis, method2test)
    plt.ylim = [0, 1]
    plt.title("Evaluation of lsd methods")
    plt.legend()
    plt.show()
