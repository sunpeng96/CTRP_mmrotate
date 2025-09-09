# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile, LoadAnnotations

from ..builder import ROTATED_PIPELINES

@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results

# @ROTATED_PIPELINES.register_module()
# class LoadMaskAnnotations(LoadAnnotations):
#
#     def __init__(self,
#                  with_bbox=True,
#                  with_label=True,
#                  with_mask_reason=True,
#                  with_mask=False,
#                  with_seg=False,
#                  poly2mask=True,
#                  denorm_bbox=False,
#                  file_client_args=dict(backend='disk')
#                  ):
#         super(LoadMaskAnnotations, self).__init__(
#             with_bbox=with_bbox,
#             with_label=with_label,
#             with_mask=with_mask,
#             with_seg=with_seg,
#             poly2mask=poly2mask,
#             denorm_bbox=denorm_bbox,
#             file_client_args=file_client_args
#         )
#         self.with_mask_reason = with_mask_reason
#
#     def _load_bboxes(self, results):
#         """Private function to load bounding box annotations.
#
#         Args:
#             results (dict): Result dict from :obj:`mmdet.CustomDataset`.
#
#         Returns:
#             dict: The dict contains loaded bounding box annotations.
#         """
#
#         ann_info = results['ann_info']
#         results['gt_bboxes'] = ann_info['bboxes'].copy()
#
#         if self.denorm_bbox:
#             bbox_num = results['gt_bboxes'].shape[0]
#             if bbox_num != 0:
#                 h, w = results['img_shape'][:2]
#                 results['gt_bboxes'][:, 0::2] *= w
#                 results['gt_bboxes'][:, 1::2] *= h
#
#         gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
#         if gt_bboxes_ignore is not None:
#             results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
#             results['bbox_fields'].append('gt_bboxes_ignore')
#         results['bbox_fields'].append('gt_bboxes')
#
#         gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
#         if gt_is_group_ofs is not None:
#             results['gt_is_group_ofs'] = gt_is_group_ofs.copy()
#
#         return results
#
#     def _load_labels(self, results):
#         """Private function to load label annotations.
#
#         Args:
#             results (dict): Result dict from :obj:`mmdet.CustomDataset`.
#
#         Returns:
#             dict: The dict contains loaded label annotations.
#         """
#         results['gt_labels'] = results['ann_info']['labels'].copy()
#
#         return results