# Tablet-Detection-with-Object-Detection
| 모델 | 계열 | 정확도 (mAP@0.5) | 속도 (FPS, RTX3060 기준) | 특징 요약 |
|:------|:------|:------:|:------:|:------|
| **VGGNet-16** | CNN Backbone | 🔹 **55 – 60 %** | 🐢 20 – 25 | 단순 구조, 학습 안정적이지만 표현력 부족 |
| **ResNet-50** | CNN Backbone | 🔹 **65 – 70 %** | ⚡ 45 – 50 | 기본 백본으로 우수, 최신 대비 효율 낮음 |
| **YOLOv8** | 1-Stage Detector | 🔹 **78 – 82 %** | ⚡ 70 + | 매우 빠름, 실시간 응용 적합 |
| **YOLOv9** | 1-Stage Hybrid | 🔹 **85 – 88 %** | ⚡ 60 + | NMS-free 구조, 정확도 향상 |
| **RT-DETR** | Transformer-based | 🔹 **90 – 93 %** | ⚡ 40 – 45 | YOLO급 속도 + DETR급 정밀도 |
| **Cascade R-CNN** | 2-Stage Detector | 🔹 **90 – 92 %** | ⚙️ 15 – 20 | Stage-wise 정밀 검출, 작은 객체에 강함 |
| **ConvNeXt + Cascade R-CNN** | Hybrid CNN | 🔹 **92 – 94 %** | ⚙️ 18 – 20 | 최신 백본 + 고정밀 탐지, 연구/산업용 최고 |
| **EfficientDet-D3** | Hybrid CNN | 🔹 **83 – 87 %** | ⚙️ 30 – 35 | 경량·효율적, 중간급 정확도 |



| 모델 | 논문 공식 URL | 공식 GitHub 구현 URL | PyTorch 공식 구현 URL |
|------|---------------|----------------------|------------------------|
| **VGGNet-16** | [Very Deep Convolutional Networks for Large-Scale Image Recognition (arXiv:1409.1556)](https://arxiv.org/abs/1409.1556) | [Oxford VGG Group Research Page](https://www.robots.ox.ac.uk/~vgg/research/very_deep/) | [torchvision.models.vgg16](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html) |
| **ResNet-50** | [Deep Residual Learning for Image Recognition (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385) | [Kaiming He: deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks) | [torchvision.models.resnet50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) |
| **YOLOv8** | (논문 미공개, Ultralytics 내부 문서) | [Ultralytics YOLOv8 Repository](https://github.com/ultralytics/ultralytics) | [Ultralytics YOLO Docs](https://docs.ultralytics.com/models/yolov8) |
| **YOLOv9** | (비공식 논문 단계, Ultralytics 비교 문서 참조) | [Ultralytics YOLOv9 Info](https://docs.ultralytics.com/compare/efficientdet-vs-yolov9/) | — |
| **RT-DETR** | [DETRs Beat YOLOs on Real-time Object Detection (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html) | [PaddlePaddle RT-DETR Repository](https://github.com/lyuwenyu/RT-DETR) | — |
| **Cascade R-CNN** | [Cascade R-CNN: Delving into High Quality Object Detection (arXiv:1906.09756)](https://arxiv.org/abs/1906.09756) | [Detectron Cascade R-CNN (official implementation)](https://github.com/zhaoweicai/Detectron-Cascade-RCNN) | [OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection) |
| **ConvNeXt + Cascade R-CNN** | [A ConvNet for the 2020s (arXiv:2201.03545)](https://arxiv.org/abs/2201.03545) + [Cascade R-CNN (arXiv:1906.09756)](https://arxiv.org/abs/1906.09756) | [ConvNeXt Official Repository](https://github.com/facebookresearch/ConvNeXt) + [MMDetection](https://github.com/open-mmlab/mmdetection) | [torchvision.models.convnext](https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_base.html) |
| **EfficientDet-D3** | [EfficientDet: Scalable and Efficient Object Detection (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf) | [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | — |