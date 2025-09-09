# -*- coding: utf-8 -*-
# @Time    : 11/15/2023
# @Author  : Peng Sun
# @FileName: ctrp_orcnn.py
# @Software: PyCharm

import mmcv
import numpy as np

from .image import draw_rbboxes, _get_adaptive_scales

import matplotlib.pyplot as plt
from mmdet.core.visualization import palette_val
from mmrotate.core.visualization.palette import get_palette
from mmdet.core.visualization.image import draw_labels, draw_masks



EPS = 1e-2

def imgshow_reason_rbboxes(img,
                           mask_bboxes, mask_labels,
                           reason_bboxes, reason_labels,
                           class_names=None, score_thr=0,
                           bbox_color='green', text_color='green',
                           thickness=2, font_size=13,
                           win_name='', show=True,
                           wait_time=0, out_file=None):

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert reason_bboxes is not None and reason_bboxes.shape[1] == 6
        renson_scores = reason_bboxes[:, -1]
        inds = renson_scores > score_thr
        reason_bboxes = reason_bboxes[inds, :]
        reason_labels = reason_labels[inds]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(reason_labels) if len(reason_labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[reason_label] for reason_label in reason_labels]

    # num_bboxes = 0
    if mask_bboxes is not None:
        num_bboxes = mask_bboxes.shape[0]

        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[mask_label] for mask_label in mask_labels[:num_bboxes]]
        draw_rbboxes(ax, mask_bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = reason_bboxes[:, :2].astype(np.int32) + thickness
        areas = reason_bboxes[:, 2] * reason_bboxes[:, 3]
        scales = _get_adaptive_scales(areas)
        scores = reason_bboxes[:, 5] if reason_bboxes.shape[1] == 6 else None

        draw_labels(
            ax,
            reason_labels[:num_bboxes],
            positions,
            scores=scores,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img

