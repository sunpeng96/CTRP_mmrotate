# -*- coding: utf-8 -*-
# @Time    : 02/22/2024
# @Author  : Peng Sun
# @FileName: ctrp_mask_utils.py
# @Software: PyCharm

import cv2
import numpy as np

from mmrotate.core.bbox import poly2obb_np

def mask_detector(img):
    """
    Used to detect the position of mask instances, based on Opencv.
    Args:
        img (Tensor): shape (N, C, H, W) encoding input images.
            Typically these should be mean centered and std scaled.

    Returns:
        mask_result_list (List[Numpy]): in (x, y, w, h, theta) format.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度图像
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY) # 应用阈值处理
    # adaptMEAN = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=5, C=3)
    # adaptGAUS = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=5, C=3)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_bbox_pred = []
    if len(contours) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    else:
        polys = []
        for contour in contours:
            area = cv2.contourArea(contour)  # 计算点集的面积, 筛选掉面积较小的点集
            peri = cv2.arcLength(contour, True)  # 计算周长, 用于 approxPolyDP的限制条件.
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # 估计点集的形状, 筛选出形状为 4的矩形区域

            filter = (area >= 70 and len(approx) == 4)  # 制作筛选器, 筛选掉面积较小(<70)且不是矩形区域(!=4)的点集
            if filter:  # 通过过滤器的点集进行接下来的操作
                rect = cv2.minAreaRect(contour)  # 获取最小旋转矩形
                poly = cv2.boxPoints(rect)  # 得到多边形
                polys.append(poly)
                obb = poly2obb_np(poly.reshape(-1), version='le135')  # 得到(x, y, w, h, theta)的形式
                mask_bbox_pred.append(obb)

    if len(mask_bbox_pred) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    else:
        mask_result = np.stack(mask_bbox_pred, axis=0)
        confidence = np.ones(mask_result.shape[0]) * 0.85

        return np.concatenate([mask_result, confidence[:, None]], 1)

def cloud_detector(img):

    template = cv2.imread('mmrotate/models/utils/template.png')
    search = img
    template_edge = cv2.GaussianBlur(template, (5, 5), 0)
    template_edge = cv2.Canny(template_edge, 10, 200, apertureSize=3)
    search_edge = cv2.GaussianBlur(search, (5, 5), 0)

    (h1, w1) = search_edge.shape[:2]
    search_edge = cv2.Canny(search_edge, 10, 180, apertureSize=3)
    serch_ROIPart = search_edge[50:h1 - 50, 50:w1 - 50]  # 裁剪图像

    match_points = RatationMatch(template_edge, serch_ROIPart)
    (height, width) = template_edge.shape[:2]
    cx, cy = match_points['point'][0] + width / 2, match_points['point'][1] + height / 2

    mask_result = np.array([cx, cy, width, height, np.pi / 2]).reshape(-1, 5)

    confidence = np.ones(mask_result.shape[0]) * 0.85

    return np.concatenate([mask_result, confidence[:, None]], 1)

# 图片旋转函数
def ImageRotate(img, angle):  # img:输入图片；newIm：输出图片；angle：旋转角度(°)
    height, width = img.shape[:2]  # 输入(H,W,C)，取 H，W 的值
    center = (width // 2, height // 2)  # 绕图片中心进行旋转
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_rotation = cv2.warpAffine(img, M, (width, height))

    return image_rotation

# 取圆形ROI区域函数：具体实现功能为输入原图，取原图最大可能的原型区域输出
def ExtractCRoI(src):
    dst = np.zeros(src.shape, np.uint8)  # 感兴趣区域ROI
    mask = np.zeros(src.shape, dtype='uint8')  # 感兴趣区域ROI
    (h, w) = mask.shape[:2]
    (cX, cY) = (w // 2, h // 2)  # 是向下取整
    radius = int(min(h, w) / 2)
    cv2.circle(mask, (cX, cY), radius, (255, 255, 255), -1)
    # 以下是copyTo的算法原理：
    # 先遍历每行每列（如果不是灰度图还需遍历通道，可以事先把mask图转为灰度图）
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            # 如果掩图的像素不等于0，则dst(x,y) = scr(x,y)
            if mask[row, col] != 0:
                # dst_image和scr_Image一定要高宽通道数都相同，否则会报错
                dst[row, col] = src[row, col]
                # 如果掩图的像素等于0，则dst(x,y) = 0
            elif mask[row, col] == 0:
                dst[row, col] = 0
    return dst

# 金字塔下采样
def ImagePyrDown(image, NumLevels):
    for i in range(NumLevels):
        image = cv2.pyrDown(image)  # pyrDown下采样
    return image

# 旋转匹配函数（输入参数分别为模板图像、待匹配图像）
def RatationMatch(template_image, search_image):

    search_tmp = ImagePyrDown(search_image, 3)
    template_tmp = ImagePyrDown(template_image, 3)

    newIm = ExtractCRoI(template_tmp)
    # 使用matchTemplate对原始灰度图像和图像模板进行匹配
    res = cv2.matchTemplate(search_tmp, newIm, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
    location = min_indx
    temp = min_val
    angle = 0  # 当前旋转角度记录为0

    # 以步长为5进行第一次粗循环匹配
    for i in range(-180, 181, 5):
        newIm = ImageRotate(template_tmp, i)
        newIm = ExtractCRoI(newIm)
        res = cv2.matchTemplate(search_tmp, newIm, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
        if min_val < temp:
            location = min_indx
            temp = min_val
            angle = i

    # 在当前最优匹配角度周围10的区间以1为步长循环进行循环匹配计算
    for j in range(angle - 5, angle + 6):
        newIm = ImageRotate(template_tmp, j)
        newIm = ExtractCRoI(newIm)
        res = cv2.matchTemplate(search_tmp, newIm, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
        if min_val < temp:
            location = min_indx
            temp = min_val
            angle = j

    # 在当前最优匹配角度周围2的区间以0.1为步长进行循环匹配计算
    k_angle = angle - 0.9
    for k in range(0, 19):
        k_angle = k_angle + 0.1
        newIm = ImageRotate(template_tmp, k_angle)
        newIm = ExtractCRoI(newIm)
        res = cv2.matchTemplate(search_tmp, newIm, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
        if min_val < temp:
            location = min_indx
            temp = min_val
            angle = k_angle

    # 用下采样前的图片来进行精匹配计算
    k_angle = angle - 0.1
    newIm = ImageRotate(template_image, k_angle)
    newIm = ExtractCRoI(newIm)
    res = cv2.matchTemplate(search_image, newIm, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
    location = max_indx
    temp = max_val
    angle = k_angle
    for k in range(1, 3):
        k_angle = k_angle + 0.1
        newIm = ImageRotate(template_image, k_angle)
        newIm = ExtractCRoI(newIm)
        res = cv2.matchTemplate(search_image, newIm, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
        if max_val > temp:
            location = max_indx
            temp = max_val
            angle = k_angle

    location_x = location[0] + 50
    location_y = location[1] + 50

    # 前面得到的旋转角度是匹配时模板图像旋转的角度，后面需要的角度值是待检测图像应该旋转的角度值，故需要做相反数变换
    angle = -angle
    match_point = {'angle': angle, 'point': (location_x, location_y)}

    return match_point