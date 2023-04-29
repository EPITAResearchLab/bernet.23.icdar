
import glob
import subprocess
import shutil
import os
import time
import re
from pathlib import Path

# Widgets
import ipywidgets as widgets
from IPython.display import display

# Display image
import cv2
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import gray2rgb

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
# Dataset
from lsd_path_dataset import dataset_folder
# Evaluation Vector
sys.path.append(f'{root_dir}/line_detection_evaluation/vector/linelet_modified')
from line_detection_evaluation.vector.linelet_modified.utils import *
from line_detection_evaluation.vector.linelet_modified.evaluate_line_segment import evaluate_line_segment_complete as elsv_lm
from line_detection_evaluation.vector.linelet_python.evaluate_line_segment import evaluate_line_segment_complete as elsv_l
# Evaluation Pixel
from line_detection_evaluation.pixel.icdar_2013.evaluate_staff_line import icdar_2013_evaluate_pixel
from line_detection_evaluation.pixel.icdar_2011.evaluate_staff_line import icdar_2011_evaluate_pixel
from line_detection_evaluation.pixel.coco.eval_coco import eval_coco_path, eval_coco_path_score


from typing import List, Tuple

# Tuning
TUNING_OUT = "tuning_out/"
if not os.path.isdir(TUNING_OUT):
    os.mkdir(TUNING_OUT)

OUT_IMG = os.path.join(TUNING_OUT, "out.png")
OUT_CSV = os.path.join(TUNING_OUT, "out.csv")

BIN_FOLDER = "../bin/"
ocv_cli = os.listdir(BIN_FOLDER)


# lsd_naive_determine_angle
folder_path = 'lsd_naive_determine_angle'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

image_original_filename = 'image_original.png'
csv_out = 'out.csv'

image_original_path_angle = os.path.join(folder_path, image_original_filename)
csv_out_path_angle = os.path.join(folder_path, csv_out)

# lsd_remove_line
folder_path = 'lsd_remove_line_in'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

image_original_filename: str = 'image_original.png'
image_rgb_filename: str = 'image_rgb.png'
superposition_filename: str = 'superposition.csv'
label_dict_filename: str = 'label_dict.csv'

image_original_path: Path = os.path.join(folder_path, image_original_filename)
image_rgb_path: Path = os.path.join(folder_path, image_rgb_filename)
superposition_path: Path = os.path.join(folder_path, superposition_filename)
label_dict_path: Path = os.path.join(folder_path, label_dict_filename)


folder_out_path = 'lsd_remove_line_out'

def get_w_eval() -> List[widgets.Widget]:
    w_eval_thresh_distance = widgets.BoundedFloatText(
        value=10,
        min=0,
        max=100,
        step=1,
        description='threshold distance',
        disabled=False
    )
    w_eval_thresh_angle = widgets.BoundedFloatText(
        value=3.14 * 5 / 180,
        min=0,
        max=90,
        step=1,
        description='threshold angle',
        disabled=False
    )
    w_eval_thresh_ratio = widgets.BoundedFloatText(
        value=0.75,
        min=0,
        max=1,
        step=0.1,
        description='threshold ratio',
        disabled=False
    )
    w_eval_thresh_length = widgets.BoundedIntText(
        value=0,
        min=0,
        max=999999,
        step=1,
        description='threshold length',
        disabled=False
    )

    return [w_eval_thresh_distance,
            w_eval_thresh_angle, w_eval_thresh_ratio, w_eval_thresh_length]


def get_train_dataset() -> List[str]:
    if dataset_folder is None:
        return []

    inp = []
    for dataset in ["maps", "music_sheets", "trade_directories"]:
        inp += list(glob.glob(os.path.join(dataset_folder,
                    dataset, "train", "input", "**")))

    return inp


def ocv_widget_generation(name: str) -> List[widgets.Widget]:
    """
    Generate a list of widgets for a given binary
    :param name: Name of the binary
    :return: List of widgets
    """
    cmd = [name, "-h"]
    # Redirect stdout in PIPE
    sub = subprocess.run(cmd, stdout=subprocess.PIPE)
    s = sub.stdout.decode('utf-8')  # Get Content of stdout
    tokens = re.split(" |\n\t|,|\t|\n", s)  # Remove relevent char
    maped = list(filter(lambda x: ("--" in x or "value" in x) and not any(
        y in x for y in ["--input", "output", "--reg", ".png", ".csv"]), tokens))

    linked = [(maped[i][maped[i].find("--") + 2:], maped[i + 1][7:-1])
              for i in range(0, len(maped) - (len(maped) % 2), 2)]
    # Remove the tuple containing of "help"
    removed_help = list(filter(lambda x: not "help" in x[0], linked))

    def get_widgets(x):
        name = x[0]
        value = x[1]
        if value in ["true", "false"]:
            w = widgets.IntSlider(
                min=0,
                max=1,
                value=1 if value == "true" else 0,
                step=1,
                description=name,
                disabled=False,
                indent=False
            )
        elif value.isnumeric() or all([i.isnumeric() for i in value.split('.', 1)]):
            v = float(value)
            w = widgets.BoundedFloatText(
                value=v,
                min=0,
                max=100000,
                step=1,
                description=name,
                disabled=False
            )
        else:
            w = None
            print(f"Error to handle : value {value} - name {name}")

        return w

    all_wids = list(map(get_widgets, removed_help))

    w_to = list(filter(lambda x: x.description == 'type_out', all_wids))
    w_segout = None if len(w_to) == 0 else w_to[0]

    wids, adv_wids = [], []
    for w in all_wids:
        if 'adv' in w.description:
            adv_wids.append(w)
        else:
            wids.append(w)
    return wids, adv_wids, w_segout


def any_of_list_in_word(word: str, l: List[str]) -> bool:
    """
    Check if any of the word in l is in word
    :param word: Word to check
    :param l: List of word to check
    :return: True if any of the word in l is in word
    """
    return any(w in word for w in l)


def get_out_filename(w_segout: widgets.Widget) -> str:
    """
    Get the output filename depending on the type of output
    :param w_segout: Widget of the type of output
    :return: Output filename
    """
    if w_segout is None or w_segout.value == 0:
        return OUT_CSV
    else:
        return OUT_IMG


def build_cmd(name: str, inp: str, out: str, w_args: List[widgets.Widget]) -> List[str]:
    """
    Build the command line for a given binary
    :param name: Name of the binary
    :param inp: Input file
    :param out: Output file
    :param w_args: List of widgets representing the arguments of the binary
    :return: List of string representing the command line
    """
    ret = [name] + [f"--{w.description}={w.value if not (w.__class__ is widgets.IntSlider) else ('true' if w.value == 1 else 'false')}" for w in w_args] + [
        "-i=" + inp] + ["-o=" + out]
    return ret


def vector_evaluation(T: str, P: str, w_evals: List[widgets.Widget], img_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the result of a binary using the vector evaluation
    :param T: Ground Truth file
    :param P: Prediction file
    :param w_evals: List of widgets representing the evaluation parameters
    :param img_shape: Shape of the image
    :return: Tuple containing the image with the prediction, the image with the ground truth, the image with the evaluation
    """
    print(f'args :\n - T : {T}\n - P : {P}')

    height, width = img_shape
    pr_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    pre_lines = read_csv(P).astype(int)
    for line in pre_lines:
        x0, y0, x1, y1 = line[0], line[1], line[2], line[3]
        rd_color = (rd.randint(0, 254), rd.randint(0, 254), rd.randint(0, 254))
        cv2.line(pr_img, (x0, y0), (x1, y1), rd_color)

    if not T or not os.path.exists(T):
        print('No Ground Truth file!')
        return pr_img, None, None

    params = eval_param_struct(
        w_evals[0].value, w_evals[1].value, w_evals[2].value, min_length=w_evals[3].value)

    pr, re, iou, fsc = elsv_l(T, P, params)
    print(
        f'LineLet evaluation\n Precision : {pr}\n Recall : {re}\n IoU : {iou}\n F1-score : {fsc}')

    params.split_penalized = False
    pr, re, iou, fsc = elsv_lm(T, P, params)
    print(
        f'LineLet MODIFIED evaluation\n Precision : {pr}\n Recall : {re}\n IoU : {iou}\n F1-score : {fsc}')

    params.split_penalized = True
    pr, re, iou, fsc = elsv_lm(T, P, params)
    print(
        f'LineLet MODIFIED evaluation (split penalized)\n Precision : {pr}\n Recall : {re}\n IoU : {iou}\n F1-score : {fsc}')

    height, width = img_shape
    ref_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    pr_re_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    ref_lines = read_csv(T).astype(int)

    if ref_lines.shape == (4, ):
        ref_lines = ref_lines.reshape(1, 4)
    if pre_lines.shape == (4, ):
        pre_lines = pre_lines.reshape(1, 4)

    rect = ref_lines[0]
    rx0, ry0, w, h = rect[0], rect[1], rect[2], rect[3]
    cv2.rectangle(pr_re_img, (rx0, ry0), (rx0 + w, ry0 + h), (0, 0, 0))
    for line in ref_lines[1:]:
        x0, y0, x1, y1 = line[0], line[1], line[2], line[3]
        cv2.line(ref_img, (x0, y0), (x1, y1), (0, 255, 0))
        cv2.line(pr_re_img, (x0, y0), (x1, y1), (0, 255, 0))
    for line in pre_lines:
        x0, y0, x1, y1 = line[0], line[1], line[2], line[3]
        cv2.line(pr_re_img, (x0, y0), (x1, y1), (0, 0, 0))

    return pr_img, ref_img, pr_re_img


def pixel_evaluation(gt: str, pr: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the result of a binary using the pixel evaluation
    :param gt: Ground Truth file
    :param pr: Prediction file
    :return: Tuple containing the image with the prediction, the image with the ground truth, the image with the evaluation
    """
    print(f'args :\n - T : {gt}\n - P : {pr}')
    pr_img = cv2.imread(pr)
    if not gt or not os.path.exists(gt) or gt == pr:
        print('No Ground Truth file!')
        return pr_img, None, None

    icdar2011_score = icdar_2011_evaluate_pixel(gt, pr)
    print(f'ICDAR 2011 (classification)\n Score = {icdar2011_score}')

    icdar2013_score = icdar_2013_evaluate_pixel(gt, pr)
    print(f'ICDAR 2013 (classification)\n F1-score = {icdar2013_score}')

    PQ, SQ, RQ, _, *pr_re_img = eval_coco_path(gt, pr)
    print(
        f'COCO (labelisation)\n Panoptic Quality = {PQ}\n Segmentation Quality = {SQ}\n Recognition Quality = {RQ}')

    return pr_img, gray2rgb(imread(gt)), pr_re_img


def multiple_choice_widget(opts: List[str], desc: str) -> widgets.Dropdown:
    """
    Create a widget to choose between multiple options
    :param opts: List of options
    :param desc: Description of the widget
    :return: Dropdown widget
    """
    return widgets.Dropdown(
        options=[(opts[i], i) for i in range(len(opts))],
        value=0,
        description=desc,
    )


def get_inputs_widget() -> widgets.Dropdown:
    """
    Create a widget to choose between multiple inputs
    :return: Dropdown widget
    """
    inp = []
    inp += list(glob.glob("image/**"))
    inp += get_train_dataset()
    return multiple_choice_widget(inp, "input")


def tune():
    """
    Tune the parameters of the multiple binaries linear object detector: Pylene, LSD, Hough, Canny, Cannyline, Edline, Elsed
    Binaries should posses a opencv CLI interface with the --help option.

    Depending on the output type, the evaluation will be different : pixel or vectorial

    Pixel evaluation: ICDAR 2011, ICDAR 2013, COCO Panoptic
    Vectorial evaluation: Linelet evaluation (translation from Matlab), LEMS, LEM

    Images are updated when:
    - the input image is changed
    - parameters are changed
    - evaluation parameters are changed (for vectorial evaluation only)
    """
    w_output = widgets.Output()
    w_choice = multiple_choice_widget(ocv_cli, 'LSD method:')
    w_inputs = get_inputs_widget()
    w_mv_pixel_remove_line = widgets.Button(
        description="Move to remove line",
        disabled=True,
        indent=False
    )
    w_mv_vector_angle = widgets.Button(
        description="Move to determine angle",
        disabled=True,
        indent=False
    )
    w_save_vector = widgets.Button(
        description="Save vector output",
        disabled=True,
        indent=False
    )

    tab = widgets.Tab()
    titles = ['original', 'over', 'alone', 'ref and detected', 'ref']
    tab.children = [widgets.Output() for i in range(len(titles))]
    for i in range(len(titles)):
        tab.set_title(i, titles[i])

    def tunne_me(change):
        w_output.clear_output()
        with w_output:
            # Name of the binary used
            name = os.path.join(BIN_FOLDER, w_choice.options[w_choice.value][0])

            # Modify global widgets
            w_arguments_wid, w_arguments_wid_adv, w_segout = ocv_widget_generation(
                name)
            w_output_exec = widgets.Output()

            # Evaluation
            w_output_eval = widgets.Output()
            w_evals = get_w_eval()

            def eval_curr(change, t=-1):
                w_output_eval.clear_output()
                with w_output_eval:
                    print(f"Duration : {t}")
                    input_filename = w_inputs.options[w_inputs.value][0]
                    tested = get_out_filename(w_segout)
                    if w_segout is None or w_segout.value == 0:  # Vector
                        gt_csv = (input_filename
                                  .replace('.png', '.csv').replace('.pgm', '.csv')
                                  .replace('image', 'csv')
                                  .replace('input', 'ground truth'))
                        pre, ref, ref_and_det = vector_evaluation(
                            gt_csv, tested, w_evals, imread(input_filename).shape[:2])

                    else:  # Pixel
                        gt = input_filename.replace('input', 'ground truth')
                        if gt == input_filename:
                            gt = None
                        pre, ref, ref_and_det = pixel_evaluation(gt, tested)
                        if ref is not None and ref.shape[2] == 4:
                            ref = ref[:, :, :, 0]

                return pre, ref, ref_and_det

            def on_value_change(change, dump=False):
                inp = w_inputs.options[w_inputs.value][0]
                out_filename = get_out_filename(w_segout)
                cmd = build_cmd(name, inp, out_filename,
                                w_arguments_wid + w_arguments_wid_adv)

                t0 = time.time()
                s = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                t1 = time.time()

                def display_image(w_out_id, image):
                    tab.children[w_out_id].clear_output()
                    if image is None:
                        return
                    with tab.children[w_out_id]:
                        plt.figure(w_out_id).clear()
                        if type(image) != list:
                            plt.imshow(image)
                        else:
                            f, [ax1, ax2] = plt.subplots(
                                2, 1, figsize=(12, 10))
                            ax1.imshow(image[0])
                            ax1.set_title("Precision")
                            ax2.imshow(image[1])
                            ax2.set_title("Recall")
                        plt.show()

                # Get images
                met_in = cv2.imread(inp, cv2.IMREAD_COLOR)
                met_out, met_ref, met_ref_and_det = eval_curr(None, t=(t1-t0))
                met_over = np.where(
                    met_out == [255, 255, 255], met_in, met_out)

                # Display images
                images = [met_in, met_over, met_out, met_ref_and_det, met_ref]
                for i in range(len(images)):
                    display_image(i, images[i])

                # Enable moving to remove_line input folder
                w_mv_pixel_remove_line.disabled = w_segout is not None and w_segout.value != 1
                w_mv_vector_angle.disabled = not w_mv_pixel_remove_line.disabled
                w_save_vector.disabled = not w_mv_pixel_remove_line.disabled

                def save_vector(event):
                    with w_output_exec:
                        print(f"Saving vector")
                    cv2.imwrite("ori_in.png", met_in)
                    cv2.imwrite("vector_out.png", met_out)
                w_save_vector._click_handlers.callbacks = []
                w_save_vector.on_click(save_vector)

                with w_output_exec:
                    err = s.stderr.decode('utf-8')
                    if err:
                        print(err)

            for w in w_arguments_wid:
                w.observe(on_value_change, names="value")
            for w in w_arguments_wid_adv:
                w.observe(on_value_change, names="value")
            w_inputs.observe(on_value_change, names="value")

            def mv_pixel_remove_line(event):
                with w_output_exec:
                    print(f"Moving to remove_line folder")
                shutil.copy(
                    w_inputs.options[w_inputs.value][0], image_original_path)
                shutil.copy(get_out_filename(w_segout), image_rgb_path)
                shutil.copy('pixel_label_dict.csv', label_dict_path)
                shutil.copy('superposition.csv', superposition_path)
            w_mv_pixel_remove_line.on_click(mv_pixel_remove_line)
            w_mv_pixel_remove_line.disabled = True

            def mv_vector_angle(event):
                with w_output_exec:
                    print(f"Moving to angle folder")
                shutil.copy(
                    w_inputs.options[w_inputs.value][0], image_original_path_angle)
                shutil.copy(get_out_filename(w_segout), csv_out_path_angle)
            w_mv_vector_angle.on_click(mv_vector_angle)
            w_mv_vector_angle.disabled = True

            for w_eval in w_evals:
                w_eval.observe(eval_curr, names='value')
            box_bin_eval = widgets.Box(children=w_evals)

            box_bin = widgets.Box(children=w_arguments_wid)
            box_bin.layout.display = 'flex'
            box_bin.layout.flex_flow = 'wrap'
            box_bin.layout.align_items = 'stretch'

            w_disp_adv = widgets.IntSlider(
                value=0, min=0, max=1, step=1, description='Advanced')

            box_bin_adv = widgets.Box(children=w_arguments_wid_adv)
            box_bin_adv.layout.display = 'flex'
            box_bin_adv.layout.flex_flow = 'wrap'
            box_bin_adv.layout.align_items = 'stretch'

            def f_disp_adv(change=None):
                w_output.clear_output()
                # w_output_eval.clear_output()
                # w_output_exec.clear_output()
                with w_output:
                    l = [box_bin]
                    if len(w_arguments_wid_adv) > 0:
                        l.append(w_disp_adv)
                    if w_disp_adv.value == 1:
                        l.append(box_bin_adv)
                    l += [box_bin_eval, w_output_eval, w_output_exec, tab]
                    for w in l:
                        display(w)

            w_disp_adv.observe(f_disp_adv, names='value')

            f_disp_adv()

    w_choice.observe(tunne_me, names="value")

    display(widgets.Box(children=[
            w_choice, w_inputs, w_mv_pixel_remove_line, w_mv_vector_angle, w_save_vector]))
    display(w_output)
