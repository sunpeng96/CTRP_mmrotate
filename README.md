# Completing Missing Entities: Exploring Consistency Reasoning for Oriented Object Detection
Peng Sun, Yongbin Zheng, Wanying Xu, Jian Li, and Jiansong Yang

## 🌟 Introduction
This is the official implementation of the paper: **Completing Missing Entities: Exploring Consistency Reasoning for Remote Sensing Object Detection**, which is implemented on [MMrotate](https://github.com/open-mmlab/mmrotate).
<img width="1000" height="230" alt="statistics" src="https://github.com/user-attachments/assets/1f09b897-4802-48a4-b34b-96c5d22f0ada" />

<img width="1000" height="360" alt="overview" src="https://github.com/user-attachments/assets/5ca4cec3-eecd-47a5-9e3a-166bbd78625c" />

---

## 🌟 MMRotate官方使用文档
[English](/README-EN.md) | [简体中文](/README-CN.md)

---

## 🌟 Code implementation
We provide a demo to verify the experimental results, and the complete code will be released after the paper is accepted. 

You can use the program in the demo to carry out experiments. where [checkpoint](https://pan.baidu.com/s/1TeZONH2360V73pP3aOKHrQ?pwd=qwer) is used for remote sensing image object detection tasks, and [checkpoint](https://pan.baidu.com/s/1RlZdSoK3_AEEQbfFU7r14w?pwd=qwer) is used for occluded object detection tasks.

---

## 🌟 Qualitative Results
### Remote Sensing Object Detection Task
<img width="1000" height="480" alt="remote sensing object detection yask" src="https://github.com/user-attachments/assets/7ffa06bf-362a-4ecc-a504-fee0247fcf23" />

### Occluded Object Detection Task
<img width="1000" height="480"  alt="occluded object detection task" src="https://github.com/user-attachments/assets/08e5a575-f91a-47f9-8c1c-bb8dd53baea9" />

## 🌟 Quantitative Results
## Benchmark
### Remote Sensing Object Detection Task
| Model | Backbone | Dataset | Lr schd |  mAP | Configs | Download |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| FR-O w/ CTRP   | R50-FPN  | DOTA   | 1x  | 75.12 | [ctrp_rotated_faster_rcnn_r50_fpn_1x_dota_le90.py](CTRP_for_Remote_Sensing_Object_Detection%2Fconfigs%2Fctrp%2Fctrp_rotated_faster_rcnn_r50_fpn_1x_dota_le90.py) | [model](https://pan.baidu.com/s/1TeZONH2360V73pP3aOKHrQ?pwd=qwer) |
| FR-O w/ CTRP   | R101-FPN | DOTA   | 1x  | 76.05 | [ctrp_rotated_faster_rcnn_r101_fpn_1x_dota_le90.py](CTRP_for_Remote_Sensing_Object_Detection%2Fconfigs%2Fctrp%2Fctrp_rotated_faster_rcnn_r101_fpn_1x_dota_le90.py) | [model](https://pan.baidu.com/s/1TeZONH2360V73pP3aOKHrQ?pwd=qwer) |
| O-RCNN w/ CTRP | R50-FPN  | DOTA   | 1x  | 77.17 | [ctrp_oriented_rcnn_r50_fpn_1x_dota_le90.py](CTRP_for_Remote_Sensing_Object_Detection%2Fconfigs%2Fctrp%2Fctrp_oriented_rcnn_r50_fpn_1x_dota_le90.py) | [model](https://pan.baidu.com/s/1TeZONH2360V73pP3aOKHrQ?pwd=qwer) |
| O-RCNN w/ CTRP | R101-FPN | DOTA   | 1x  | 77.80 | [ctrp_oriented_rcnn_r101_fpn_1x_dota_le90.py](CTRP_for_Remote_Sensing_Object_Detection%2Fconfigs%2Fctrp%2Fctrp_oriented_rcnn_r101_fpn_1x_dota_le90.py) | [model](https://pan.baidu.com/s/1TeZONH2360V73pP3aOKHrQ?pwd=qwer) |
| O-RCNN w/ CTRP | R50-FPN  | DIOR-R | 1x  | 65.53 | [ctrp_oriented_rcnn_r101_fpn_1x_dior_r_le90.py](CTRP_for_Remote_Sensing_Object_Detection%2Fconfigs%2Fctrp%2Fctrp_oriented_rcnn_r101_fpn_1x_dior_r_le90.py) | [model](https://pan.baidu.com/s/1TeZONH2360V73pP3aOKHrQ?pwd=qwer) |

### Occluded Object Detection Task
| Model | Backbone | Dataset | Lr schd |  mAP | Configs | Download |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| FR-O w/ CTRP   | R50-FPN  | Occluded DOTA   | 1x  | 78.30 | [ctrp_rotated_faster_rcnn_r50_fpn_1x_occluded_dota_le90.py](CTRP_for_Occluded_Object_Detection%2Fconfigs%2Fctrp_frcnn_o%2Fctrp_rotated_faster_rcnn_r50_fpn_1x_occluded_dota_le90.py) | [model](https://pan.baidu.com/s/1RlZdSoK3_AEEQbfFU7r14w?pwd=qwer) |
| FR-O w/ CTRP   | R101-FPN  | Occluded DOTA   | 1x  | 79.29 | [ctrp_rotated_faster_rcnn_r101_fpn_1x_occluded_dota_le90.py](CTRP_for_Occluded_Object_Detection%2Fconfigs%2Fctrp_frcnn_o%2Fctrp_rotated_faster_rcnn_r101_fpn_1x_occluded_dota_le90.py) | [model](https://pan.baidu.com/s/1RlZdSoK3_AEEQbfFU7r14w?pwd=qwer) |
| O-RCNN w/ CTRP | R50-FPN | Occluded DOTA   | 1x  | 79.76 | [ctrp_oriented_rcnn_r50_fpn_1x_occluded_dota_le90.py](CTRP_for_Occluded_Object_Detection%2Fconfigs%2Fctrp_orcnn%2Fctrp_oriented_rcnn_r50_fpn_1x_occluded_dota_le90.py) | [model](https://pan.baidu.com/s/1RlZdSoK3_AEEQbfFU7r14w?pwd=qwer) |
| O-RCNN w/ CTRP | R101-FPN | Occluded DOTA   | 1x  | 80.52 | [ctrp_oriented_rcnn_r101_fpn_1x_occluded_dota_le90.py](CTRP_for_Occluded_Object_Detection%2Fconfigs%2Fctrp_orcnn%2Fctrp_oriented_rcnn_r101_fpn_1x_occluded_dota_le90.py) | [model](https://pan.baidu.com/s/1RlZdSoK3_AEEQbfFU7r14w?pwd=qwer) |
| FR-O w/ CTRP   | R50-FPN  | Occluded DIOR-R | 1x  | 58.46 | [ctrp_rotated_faster_rcnn_r50_fpn_1x_occluded_dior_r_le90.py](CTRP_for_Occluded_Object_Detection%2Fconfigs%2Fctrp_frcnn_o%2Fctrp_rotated_faster_rcnn_r50_fpn_1x_occluded_dior_r_le90.py) | [model](https://pan.baidu.com/s/1RlZdSoK3_AEEQbfFU7r14w?pwd=qwer) |
| FR-O w/ CTRP   | R101-FPN  | Occluded DIOR-R | 1x  | 59.43 | [ctrp_rotated_faster_rcnn_r101_fpn_1x_occluded_dior_r_le90.py](CTRP_for_Occluded_Object_Detection%2Fconfigs%2Fctrp_frcnn_o%2Fctrp_rotated_faster_rcnn_r101_fpn_1x_occluded_dior_r_le90.py) | [model](https://pan.baidu.com/s/1RlZdSoK3_AEEQbfFU7r14w?pwd=qwer) |
| O-RCNN w/ CTRP | R50-FPN | Occluded DIOR-R | 1x  | 61.54 | [ctrp_oriented_rcnn_r50_fpn_1x_occluded_dior_r_le90.py](CTRP_for_Occluded_Object_Detection%2Fconfigs%2Fctrp_orcnn%2Fctrp_oriented_rcnn_r50_fpn_1x_occluded_dior_r_le90.py) | [model](https://pan.baidu.com/s/1RlZdSoK3_AEEQbfFU7r14w?pwd=qwer) |
| O-RCNN w/ CTRP | R101-FPN | Occluded DIOR-R | 1x  | 63.14 | [ctrp_oriented_rcnn_r101_fpn_1x_occluded_dior_r_le90.py](CTRP_for_Occluded_Object_Detection%2Fconfigs%2Fctrp_orcnn%2Fctrp_oriented_rcnn_r101_fpn_1x_occluded_dior_r_le90.py) | [model](https://pan.baidu.com/s/1RlZdSoK3_AEEQbfFU7r14w?pwd=qwer) |

---

## 🌟 Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@article{Li_2024_IJCV,
  title={LSKNet: A Foundation Lightweight Backbone for Remote Sensing},
  author={Li, Yuxuan and Li, Xiang and Dai, Yimain and Hou, Qibin and Liu, Li and Liu, Yongxiang and Cheng, Ming-Ming and Yang, Jian},
  journal={International Journal of Computer Vision},
  year={2024},
  doi = {https://doi.org/10.1007/s11263-024-02247-9},
  publisher={Springer}
}

@InProceedings{Li_2023_ICCV,
    author    = {Li, Yuxuan and Hou, Qibin and Zheng, Zhaohui and Cheng, Ming-Ming and Yang, Jian and Li, Xiang},
    title     = {Large Selective Kernel Network for Remote Sensing Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16794-16805}
}

@article{yuan2025strip,
  title={Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection},
  author={Yuan, Xinbin and Zheng, ZhaoHui and Li, Yuxuan and Liu, Xialei and Liu, Li and Li, Xiang and Hou, Qibin and Cheng, Ming-Ming},
  journal={arXiv preprint arXiv:2501.03775},
  year={2025}
}

```
