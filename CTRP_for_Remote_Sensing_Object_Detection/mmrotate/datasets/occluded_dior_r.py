# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial
from multiprocessing import get_context

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset
from mmcv import print_log

from .dior_r import DIOR_RDataset

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS
from collections import OrderedDict

@ROTATED_DATASETS.register_module()
class OccludedDIOR_RDataset(DIOR_RDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt',
               'bridge', 'chimney', 'Expressway-Service-area', 'Expressway-toll-station',
               'dam', 'golffield', 'groundtrackfield', 'harbor',
               'overpass', 'ship', 'stadium', 'storagetank',
               'tenniscourt', 'trainstation', 'vehicle', 'windmill', 'mask')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255), (42, 165, 42),
               (42, 42, 165), (107, 189, 183), (153, 51, 0), (0, 128, 255),
               (255, 0, 0)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(OccludedDIOR_RDataset, self).__init__(ann_file, pipeline, **kwargs)

    def get_mask_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['mask_ann']

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.jpg')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
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
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
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

                if os.path.getsize(ann_file) == 0:
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
                            label = 20 if mask > 0 else cls_map[cls_name]  # 被掩膜的实例的类别设为 15, 用于目标检测.
                            mask_label = cls_map[cls_name]  # 用于指示实例被掩膜之前的类别.

                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                            if mask is not None and mask > 0:  # 测试模式中, 实例被遮挡的情况 (mask为 1).
                                gt_mask_bboxes.append([x, y, w, h, a])
                                gt_mask_polygons.append(poly)
                                gt_mask_labels.append(mask_label)

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

    def evaluate(self,
                 results,
                 metric='mAP',
                 eval_mode='ablation',  # 'ablation' or 'sota'
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 # iou_thr=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
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
                bbox_results.append(result[0][:20]), mask_results.append(result[1])

            print('\n----------SOTA evaluation----------')
            total_annotations = []
            for i, (ann, mask_ann) in enumerate(zip(annotations, mask_annotations)):
                keep_inds = ann['labels'] != 20
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
                    dataset=self.CLASSES[:20],  # 20类目标, 无 Mask类
                    logger=logger,
                    nproc=nproc)
                eval_results['mAP'] = mean_ap
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
                    dataset=self.CLASSES[:20],  # 20类目标, 无 Mask类
                    logger=logger,
                    nproc=nproc)
                eval_results['mAP'] = mean_ap
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
                    dataset=self.CLASSES[:20],  # 20类目标, 无 Mask类
                    logger=logger,
                    nproc=nproc)
                eval_results['mAP'] = mean_ap
            else:
                raise NotImplementedError

            print('reason results for mask only:')
            # 计算关系推理网络的准确率
            if metric == 'mAP':
                assert isinstance(iou_thr, float)
                mean_ap, eval_mask = eval_rbbox_map(
                    mask_results,  # mask_results
                    mask_annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    show_summary=True,
                    dataset=self.CLASSES[:20],  # 20类目标, 无 Mask类
                    logger=logger,
                    nproc=nproc)

                eval_results['reason_mAP'] = mean_ap

            else:
                raise NotImplementedError

        return eval_results

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)

        files = [
            osp.join(out_folder, 'Task1_' + cls + '.txt')
            for cls in self.CLASSES
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    txt_element = [img_id, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(out_folder)[-1]
        with zipfile.ZipFile(
                osp.join(out_folder, target_name + '.zip'), 'w',
                zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results