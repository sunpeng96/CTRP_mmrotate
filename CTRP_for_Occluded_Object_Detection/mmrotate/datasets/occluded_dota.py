# -*- coding: utf-8 -*-
# @Time    : 12/07/2023
# @Author  : Peng Sun
# @FileName: occluded_dota.py
# @Software: PyCharm

import glob
import os
import numpy as np
import os.path as osp
from .dota import DOTADataset
from .builder import ROTATED_DATASETS
from mmrotate.core import eval_rbbox_map, poly2obb_np, obb2poly_np

@ROTATED_DATASETS.register_module()
class OccludedDOTADataset(DOTADataset):

    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool', 'helicopter', 'mask')

    PALETTE = [(244, 67, 54), (233, 30, 99),  (156, 39, 176), (103, 58, 183),
               (63, 81, 181), (33, 150, 243), (0, 188, 212),  (0, 150, 136),
               (76, 175, 80), (139, 195, 74), (205, 220, 57), (255, 235, 59),
               (255, 152, 0), (255, 87, 34),  (180, 0, 0),    (100, 100, 100)]

    # 在 DOTADataset类中, self.get_ann_info调用了 load_annotations函数,
    # 因此, 所有的标注信息都存储在 self.get_ann_info中.

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(OccludedDOTADataset, self).__init__(ann_file, pipeline, **kwargs)

    def get_mask_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['mask_ann']

    def evaluate(self,
                 results,
                 metric='mAP',
                 eval_mode='ablation',  # 'ablation' or 'sota'
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            eval_mode:
            results: two (list) bbox_results, mask_results.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """

        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        mask_annotations = [self.get_mask_ann_info(i) for i in range(len(self))]

        if eval_mode == 'sota':
            bbox_results = []; mask_results = []
            for result in results:
                bbox_results.append(result[0][:15]), mask_results.append(result[1])

            print('\n----------SOTA evaluation----------')
            total_annotations = []
            for i, (ann, mask_ann) in enumerate(zip(annotations, mask_annotations)):
                keep_inds = ann['labels'] != 15
                total_ann_labels = np.concatenate((ann['labels'][keep_inds], mask_ann['labels']), axis=0)
                total_ann_bboxes = np.concatenate((ann['bboxes'][keep_inds, :], mask_ann['bboxes']), axis=0)
                total_ann = {'labels': total_ann_labels, 'bboxes': total_ann_bboxes}
                total_annotations.append(total_ann)

            print('\nresults without reason stage:')
            eval_results = {}
            if metric == 'mAP':
                assert isinstance(iou_thr, float)
                mean_ap, _ = eval_rbbox_map(
                    bbox_results,  # bbox_results
                    total_annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    show_summary=True,
                    dataset=self.CLASSES[:15],  # 15类目标, 无 Mask类
                    logger=logger,
                    nproc=nproc)
                eval_results['detection_mAP'] = mean_ap
            else:
                raise NotImplementedError

            total_results = []
            for bbox_result, mask_result in zip(bbox_results, mask_results):
                cls_result = []
                for i in range(len(bbox_result)):
                    # if i < 15:
                    cls_bbox_res = bbox_result[i]
                    cls_mask_res = mask_result[i]

                    cls_res = np.concatenate((cls_bbox_res, cls_mask_res), axis=0)
                    # else:
                    #     cls_res = bbox_result[i]
                    cls_result.append(cls_res)
                total_results.append(cls_result)

            print('\nresults with reason stage:')
            if metric == 'mAP':
                assert isinstance(iou_thr, float)
                mean_ap, _ = eval_rbbox_map(
                    total_results,  # bbox_results
                    total_annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    show_summary=True,
                    dataset=self.CLASSES[:15],  # 15类目标, 无 Mask类
                    logger=logger,
                    nproc=nproc)
                eval_results['reason_mAP'] = mean_ap
            else:
                raise NotImplementedError

        if eval_mode == 'ablation':
            bbox_results = []; mask_results = []
            for result in results:
                bbox_results.append(result[0]), mask_results.append(result[1])

            print('\n----------Ablation evaluation----------')
            print('detection results for object and mask:')
            eval_results = {}
            if metric == 'mAP':
                assert isinstance(iou_thr, float)
                mean_ap, _ = eval_rbbox_map(
                    bbox_results,  # bbox_results
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    show_summary=True,
                    dataset=self.CLASSES,  # 16类目标, 有 Mask类
                    logger=logger,
                    nproc=nproc)
                eval_results['mAP'] = mean_ap
            else:
                raise NotImplementedError

            print('reason results for mask only:')
            # 计算关系推理网络的准确率
            if metric == 'mAP':
                assert isinstance(iou_thr, float)
                mean_ap, _ = eval_rbbox_map(
                    mask_results,  # mask_results
                    mask_annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    show_summary=True,
                    dataset=self.CLASSES[:15],  # 15类目标, 没有 Mask类
                    logger=logger,
                    nproc=nproc)

                eval_results['reason_mAP'] = mean_ap

        return eval_results

    def load_annotations(self, ann_folder):

        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files

            gt_labels: in (polys(8), class, mask, diffcults) format
                maks = 1指示实例被遮挡

            Returns:
                data_infos (list[dict]): list of image info dict where each dict
                has: 'filename', 'ann' ('bboxes', 'labels', 'bboxes_mask', 'labels_mask',
                'polygons', 'polygons_mask')
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []

        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.png')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []

                data_info['mask_ann'] = {}
                data_info['mask_ann']['bboxes'] = []
                data_info['mask_ann']['labels'] = []

                data_infos.append(data_info)

        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                # 没有被 mask的检测真值
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                data_info['mask_ann'] = {}
                # 被 mask的检测真值
                gt_mask_bboxes = []
                gt_mask_labels = []
                gt_mask_polygons = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue

                        mask = None
                        if len(bbox_info) == 10:  # 训练模式的数据导入, mask的标签是程序生成的
                            cls_name = bbox_info[8]
                            difficulty = int(bbox_info[9])
                            label = cls_map[cls_name]

                        elif len(bbox_info) == 11:  # 训练模式的数据导入, mask的标签由数据集生成
                            cls_name = bbox_info[8]
                            difficulty = int(bbox_info[10])
                            mask = int(bbox_info[9])  # 用于指示改实例是否被 mask.
                            label = 15 if mask > 0 else cls_map[cls_name]  # 被掩膜的实例的类别设为 15, 用于目标检测.
                            mask_label = cls_map[cls_name]  # 用于指示实例被掩膜之前的类别.

                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_polygons.append(poly)
                            gt_labels.append(label)

                            if mask is not None and mask > 0:  # 测试模式中, 实例被遮挡的情况 (mask为 1).
                                gt_mask_bboxes.append([x, y, w, h, a])
                                gt_mask_polygons.append(poly)
                                gt_mask_labels.append(mask_label)

                            # if mask is None or mask < 1:  # 在训练模式 (mask为 None)和测试模式 (mask为 0)时没有实例遮挡的情况.
                            #     gt_bboxes.append([x, y, w, h, a])
                            #     gt_polygons.append(poly)
                            #     gt_labels.append(label)
                            #
                            # elif mask is not None and mask > 0:  # 测试模式中, 实例被遮挡的情况 (mask为 1).
                            #     gt_bboxes_mask.append([x, y, w, h, a])
                            #     gt_polygons_mask.append(poly)
                            #     gt_labels_mask.append(mask_label)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                if gt_mask_bboxes:
                    data_info['mask_ann']['bboxes'] = np.array(
                        gt_mask_bboxes, dtype=np.float32)
                    data_info['mask_ann']['labels'] = np.array(
                        gt_mask_labels, dtype=np.int64)
                    data_info['mask_ann']['polygons'] = np.array(
                        gt_mask_polygons, dtype=np.float32)
                else:
                    data_info['mask_ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['mask_ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['mask_ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]

        return data_infos