# Foraminifera Detection with YOLOv8

This repository provides a YOLOv8-based pipeline for detecting deep-sea sessile foraminifera, performing parameter optimization, and estimating individual sizes from images.

---

## 📁 Directory Structure

```text
project/
├── train.py              # Training script for YOLOv8 on the foraminifera dataset
├── param_opt.py          # PSO-based optimization of YOLO confidence and IoU thresholds
├── final.py              # Inference & size estimation using the optimized model and thresholds
├── yolov8m.pt            # Base YOLOv8 model weights (or your pretrained weights)
├── ultralytics/          # Ultralytics YOLOv8 package (if using local copy)
└── README.md
```
---

🔧 Requirements
```text
Python 3.8+
PyTorch (CPU or GPU)
Ultralytics YOLOv8
OpenCV
NumPy
```
---

📦 Dataset

Dataset configuration file:
```text
ultralytics/cfg/datasets/foraminifera.yaml
```
## 🚀 Training

Run the training script:

```bash
python train.py
```
Best model weights:
```text
runs/detect/weights/best.pt
```
---

## 🎛 Parameter Optimization

param_opt.py performs Particle Swarm Optimization (PSO) to automatically search for the best YOLO confidence and IoU thresholds by minimizing counting errors on the foraminifera dataset.
Usage (example):
```bash
python param_opt.py
```

## 📏 Inference & Size Estimation

final.py runs inference on unlabeled images using the optimized thresholds and computes the estimated physical length of each detected individual (based on bounding box diagonal and a fixed scale of 10 cm per image width).
Usage (example):
```bash
python final.py
```

Key inputs and assumptions:
- Model weights: best.pt
- Optimized parameters loaded from:
    - test_results_11_20.pkl (best [conf, iou] from previous runs)
    - history_11_20.pkl (cost history)
- Unlabeled images directory: unlabeldata/
- Training/labelled images directory: LabelmeData/
- Image width is assumed to correspond to 10 cm in real-world scale.

Main outputs:
- cost_iteration.csv – monotonic cost curve over iterations (best-so-far)
- history_iteration.csv – raw cost history from optimization
- unlabeldata_result/ –
    - Images with drawn bounding boxes
    - YOLO-format prediction .txt files
- unlabeldata_results.csv – table summarizing each individual’s estimated length

## 📝 Notes

Make sure paths in param_opt.py and final.py (e.g. LabelmeData/, unlabeldata/) match your actual folder structure.