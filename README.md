# Completing Missing Entities: Exploring Consistency Reasoning for Oriented Object Detection
Peng Sun, Yongbin Zheng, Wanying Xu, Jian Li, and Jiansong Yang

## 🌟 Introduction
This is the official implementation of the paper: **Completing Missing Entities: Exploring Consistency Reasoning for Remote Sensing Object Detection**, which is implemented on [MMrotate](https://github.com/open-mmlab/mmrotate).
<img width="1000" height="230" alt="statistics" src="https://github.com/user-attachments/assets/5abe9465-75c2-41e2-99d8-8b33259dbcd0" />

<img width="1000" height="360" alt="overview" src="https://github.com/user-attachments/assets/e4ed0a3e-7a59-4fc1-99af-223cc08a5a51" />

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
<img width="1000" height="480" alt="remote sensing object detection yask" src="https://github.com/user-attachments/assets/8dd04967-c468-4b29-a11f-65eb6138e951" />

### Occluded Object Detection Task
<img width="1000" height="480"  alt="occluded object detection task" src="https://github.com/user-attachments/assets/a333eb2e-949a-43bb-8635-5d1d2d451335" />

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
@article{Sun2025CTRP,
  title={Completing Missing Entities: Exploring Consistency Reasoning for Oriented Object Detection},
  author={Peng, Sun and Yongbin, Zheng and Wanying, Xu and Jian, Li and Jiansong, Yang},
  journal={Submitted to IEEE TRANSACTIONS ON IMAGE PROCESSING},
  year={2025},
  publisher={IEEE}
}

```
