# -*- coding: utf-8 -*-
# @Time    : 12/20/2023
# @Author  : Peng Sun
# @FileName: random_mask_multi_process.py
# @Software: PyCharm

import os
import cv2
import torch
import math
import mmrotate.utils.dota_utils as dota_utils
from mmrotate.utils.dota_utils import GetFileFromThisRootDir
import numpy as np
from tqdm import tqdm
import os.path as osp
from shapely import affinity
from shapely.geometry import Polygon

from multiprocessing import Pool
from functools import partial

from mmrotate.core.bbox import obb2hbb

def main():

    print('Instance random mask process begins!!!')

    masking = img_mask_base(srcpath='/data/dataset/DOTA/split_1024_dota1_0_ss/test/',
                            num_process=16,
                            mode='test_mode',
                            img_format='.png',
                            mask_ratio=0.1,
                            )
    masking.random_mask()

    print('DONE')

def split_single_warp(name, split_base):
    split_base.random_mask_single(name)

class img_mask_base():
    def __init__(self,
                 srcpath,
                 num_process,
                 mode='test_mode',
                 mask_ratio=0.1,
                 img_format='.png',
                 ):
        self.img_srcpath = os.path.join(srcpath, 'images')
        self.ann_srcpath = os.path.join(srcpath, 'annfiles')

        self.img_destpath = os.path.join(srcpath, 'images_' + '{}'.format(mask_ratio) + '_mask')
        self.ann_destpath = os.path.join(srcpath, 'annfiles_' + '{}'.format(mask_ratio) + '_mask')

        mkdir_if_not_exists(self.img_destpath)
        mkdir_if_not_exists(self.ann_destpath)

        self.img_format = img_format

        self.num_process = num_process
        self.mask_ratio = mask_ratio
        self.mask_mode = mode

    def random_mask(self):
        imagelist = GetFileFromThisRootDir(self.img_srcpath)
        imagenames = [dota_utils.custombasename(x) for x in imagelist if (dota_utils.custombasename(x) != 'Thumbs')]
        num_image = len(imagenames)

        print('Start processing:')
        if self.num_process == 1:
            for i in tqdm(range(num_image)):
                img_name = imagenames[i]
                self.random_mask_single(img_name)

        else:
            pool = Pool(self.num_process)
            worker = partial(split_single_warp, split_base=self)
            pool.map(worker, imagenames)
        print('Processing end')

    def random_mask_single(self, img_name):

        infile_img = os.path.join(self.img_srcpath, img_name + self.img_format)
        infile_ann = os.path.join(self.ann_srcpath, img_name + '.txt')

        # annotations
        outfile_img = os.path.join(self.img_destpath, img_name + self.img_format)
        outfile_ann = os.path.join(self.ann_destpath, img_name + '.txt')

        # load images
        image = cv2.imread(infile_img)  # img ([ndarray]): shape (1024, 1024, 3)

        # load annotation
        with open(infile_ann, 'r') as f_in:
            lines = f_in.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            splitlines = np.array(splitlines)

            if self.mask_mode == 'train_mode':
                if len(splitlines) == 0:
                    cv2.imwrite(outfile_img, image)
                    with open(outfile_ann, 'w') as f_out:
                        pass

                elif len(splitlines) == 1:
                    splitline = np.squeeze(splitlines)
                    init_area, shift_area, splitlines, scale_ratio = region_repeat(splitline)

                    # 解码初始区域和移动区域的坐标, Notes: x和 y表示的是左下角点坐标
                    init_x0, init_y0, init_x1, init_y1 = init_area
                    shift_x0, shift_y0, shift_x1, shift_y1 = shift_area

                    roi = image[init_y0:init_y1, init_x0:init_x1, :]
                    image[shift_y0:shift_y1, shift_x0:shift_x1, :] = roi[::scale_ratio, ::scale_ratio, :]

                    mask_img, mask_ids = random_instance_mask(image, splitlines, self.mask_ratio)
                    cv2.imwrite(outfile_img, mask_img)

                    save_ann(outfile_ann, splitlines, mask_ids)

            if self.mask_mode == 'test_mode':
                if len(splitlines) < 2:  # 表示这个图像中没有实例, 或者只有一个实例, 不足以构成关系对.
                    mask_img = image
                    cv2.imwrite(outfile_img, mask_img)
                    with open(outfile_ann, 'w') as f_out:
                        out_str = ''
                        for i, line in enumerate(splitlines):
                            mask_inds = '0'
                            str_line = '{} {} {} {} {} {} {} {} {} {} {}\n'.format(line[0], line[1], line[2], line[3],
                                                                                   line[4], line[5], line[6], line[7],
                                                                                   line[8], mask_inds, line[9])
                            out_str += str_line
                        f_out.write(out_str)

                else:
                    mask_img, mask_ids = random_instance_mask(image, splitlines, self.mask_ratio)
                    cv2.imwrite(outfile_img, mask_img)

                    save_ann(outfile_ann, splitlines, mask_ids)

def save_ann(outfile_ann, lines, mask_ids):
    with open(outfile_ann, 'w') as f_out:
        out_str = ''
        for i, line in enumerate(lines):
            mask_inds = '1' if i in mask_ids else '0'
            str_line = '{} {} {} {} {} {} {} {} {} {} {}\n'.format(line[0], line[1], line[2], line[3],
                                                                   line[4], line[5], line[6], line[7],
                                                                   line[8], mask_inds, line[9])
            out_str += str_line
        f_out.write(out_str + '\n')

def region_repeat(splitline):

    poly = list(map(float, splitline[:8]))
    obb = poly2obb_np_le90(np.array(poly).astype(np.int32))
    hbb = np.array(obb2hbb(torch.tensor([obb]), version='le90').squeeze(0)).astype(int)

    gt_scale = math.sqrt(hbb[2] * hbb[3])  # 计算水平框的尺度，用于判断调整区域的尺度因子 ratio.
    if gt_scale > 0 and gt_scale <= 125:
        expand_ratio = 1.5
        scale_ratio = 1
        dist_factor = 1.5
    elif gt_scale > 256 and gt_scale <= 512:
        expand_ratio = 1
        scale_ratio = 2
        dist_factor = 2
    elif gt_scale > 512 and gt_scale <= 768:
        expand_ratio = 1
        scale_ratio = 4
        dist_factor = 2.5
    else:
        expand_ratio = 1
        scale_ratio = 8
        dist_factor = 3

    # 3. 计算初始区域的尺寸, 要注意转换为左下角点和长宽的形式, 这样便于后续计算
    gt_expand = adjust_ratio(hbb, expand_ratio)  # in (x_ctr, y_ctr, w, h, angle) format.

    init_x0 = gt_expand[0] - gt_expand[2] / 2
    init_y0 = gt_expand[1] - gt_expand[3] / 2
    init_x1 = gt_expand[0] + gt_expand[2] / 2
    init_y1 = gt_expand[1] + gt_expand[3] / 2

    init_area = []
    for crood in [init_x0, init_y0, init_x1, init_y1]:
        crood = int(crood)
        if crood < 0:
            crood = 1
        if crood > 1024:
            crood = 1023
        init_area.append(crood)

    init_xctr, init_yctr = (init_area[0] + init_area[2]) / 2, (init_area[1] + init_area[3]) / 2
    init_w, init_h = make_even(init_area[2] - init_area[0]), make_even(init_area[3] - init_area[1])

    init_area = [int(init_xctr - init_w / 2), int(init_yctr - init_h / 2),
                 int(init_xctr + init_w / 2), int(init_yctr + init_h / 2)]
    # Note: 为了便于计算, init_w, init_h 必须是偶数.

    # 4. 计算区域中心点移动的距离
    shift_dist = math.sqrt(math.pow(init_w, 2) + math.pow(init_h, 2)) / dist_factor

    # 为了保证移动后的区域不超界, 这里设定移动的方向为向图像中心移动
    delta_x, delta_y = 512 - init_xctr, 512 - init_yctr
    sqrt_xy = math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2))
    # 中心点移动的x, y轴距离的计算方法为, 首先计算初始区域的中心到图像中心的角度值，这样可以保证移动后的框在图中.
    shift_x, shift_y = int(shift_dist * (delta_x / sqrt_xy)), int(shift_dist * (delta_y / sqrt_xy))

    shift_area_xctr, shift_area_yctr = init_xctr + shift_x, init_yctr + shift_y
    shift_area_w, shift_area_h = init_w / scale_ratio, init_h / scale_ratio

    shift_area = [int(shift_area_xctr - shift_area_w / 2), int(shift_area_yctr - shift_area_h / 2),
                  int(shift_area_xctr + shift_area_w / 2), int(shift_area_yctr + shift_area_h / 2)]

    polygon = Polygon([poly[i:i + 2] for i in range(0, len(poly), 2)])
    shift_polygon = affinity.translate(polygon, xoff=shift_x, yoff=shift_y)
    shift_polygon = affinity.scale(shift_polygon, xfact=scale_ratio, yfact=scale_ratio)

    a = shift_polygon.exterior.coords[:]
    x1, y1, x2, y2, x3, y3, x4, y4 = \
        a[0][0], a[0][1], a[1][0], a[1][1], a[2][0], a[2][1], a[3][0], a[3][1]

    shift_splitline = np.array([x1, y1, x2, y2, x3, y3, x4, y4, splitline[8], int(splitline[9])])
    splitlines = np.concatenate((splitline[None, :], shift_splitline[None, :]), axis=0)

    return init_area, shift_area, splitlines, scale_ratio

def random_instance_mask(image, objlines, mask_ratio):

    assert objlines.shape[0] >= 2
    num_poly = objlines.shape[0]
    noise = np.random.rand(num_poly)

    h, w, c = image.shape

    # 当GT的数量在[2, 20)的区间的时候，取一半的GT做mask处理; 当GT>=20时, 取大于50%的一个比例的GT做mask, 这个比例作为一个可以调整的超参数.
    num_mask = int(num_poly * 0.25) if (num_poly >= 1 and num_poly < 20) else int(num_poly * mask_ratio)
    ids_shuffle = np.argsort(noise)  # ascend: small is masked

    # keep_ids = ids_shuffle[num_mask:]  # keep the first subset for keep ids
    mask_ids = ids_shuffle[:num_mask]  # keep the remaining subset for mask ids

    # Extract the keep bbox and mask bbox
    objlines_mask = objlines[mask_ids, :]

    polys_mask = objlines_mask[:, :8]
    # poly_mask_split = np.array([polys_mask[:, i:i + 2] for i in range(0, polys_mask.shape[1], 2)])

    # 设置半透明掩膜的相关参数
    alpha = 1.0
    color = (255, 255, 255)
    mask = np.zeros(image.shape, dtype=np.uint8)  # 构造一个与输入图像尺寸一致的掩膜图像
    for poly in polys_mask:
        poly_split = [poly[i:i + 2] for i in range(0, len(poly), 2)]
        poly_np = np.array(poly_split, float).astype(int)  # 将字符串数组转换为整型数组
        cv2.fillPoly(mask, [poly_np], color)  # 在 gt_mask位置填充掩膜
    mask_img = cv2.addWeighted(image, 1.0, mask, alpha, 1)
    # mask_img = cv2.bitwise_and(image, image, mask=mask)

    return mask_img, mask_ids

def mkdir_if_not_exists(path):
    if not osp.exists(path):
        os.mkdir(path)

def adjust_ratio(bboxes, ratio):
    '''
    Args:
        polys ([Tensor]): shape(n, 8)
        ratio: ratio of width and height
    Returns:
        polys ([Tensor])
    '''
    bboxes[2] *= ratio
    bboxes[3] *= ratio

    return bboxes

def make_even(num):
    if num % 8 == 0:
        return num
    else:
        return num - num % 8

def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)

    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
    # if w < 2 or h < 2:
    #     return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2

    return x, y, w, h, a

if __name__ == '__main__':
    main()