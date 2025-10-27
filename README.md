# WS-DETR: A Wavelet-Mamba and Shared Query Attention Enhanced DETR for Traffic Object Detection

**WS-DETR** is a real-time traffic object detection model built upon the **RT-DETR** framework. The model incorporates two key innovations:  
- **Wavelet-Mamba Dual Path Block (WM-Dual Block)** for improving small-object detection,  
- **Shared Query Axial Cross-Attention (SQ-ACA)** for enhancing robustness and reducing false detections in challenging environments like low-light conditions.

Both modules aim to solve critical challenges in traffic object detection, particularly small object detection and misdetections in complex traffic scenarios.

## Features
- **Real-time performance**: Designed for fast, efficient detection with high accuracy.
- **Small-object detection**: Enhanced ability to detect distant or small targets.
- **Robustness**: Effectively handles low-light conditions, occlusions, and dense traffic environments.
- **Transformer-based architecture**: Built upon the powerful DETR framework for end-to-end object detection.

## Requirements
- Python 3.9.12
- PyTorch 2.1.2
- CUDA (for GPU support)
- Additional Python packages: `torchvision`, `numpy`, `matplotlib`, `scipy`

## Train and Test
The training and testing scripts for WS-DETR will be released after the paper is officially published. Please stay tuned for the updates once the paper is available.
