# 🍱 YOLO_MFK — Food Nutrient Group Detection

ระบบ Object Detection สำหรับตรวจจับและแยกประเภทอาหารตามกลุ่มสารอาหาร ได้แก่ **โปรตีน คาร์โบไฮเดรต และผัก** โดยใช้ YOLOv5 เป็น backbone

---

## 🎯 Overview

โปรเจคนี้พัฒนาโมเดล Computer Vision ที่สามารถ

- ตรวจจับอาหารจากภาพได้แบบ real-time
- แยกประเภทอาหารตามกลุ่มสารอาหาร 3 กลุ่ม
  - 🥩 **Protein** — เนื้อสัตว์ ไข่ ถั่ว
  - 🍚 **Carbohydrate** — ข้าว แป้ง เส้น
  - 🥦 **Vegetable** — ผักทุกชนิด
- รองรับทั้ง Detection และ Segmentation

---

## 🗂️ Project Structure

```
YOLO_MFK/
├── data/              # Dataset และ config files
├── models/            # Model architecture
├── classify/          # Classification module
├── segment/           # Segmentation module
├── runs/              # Training results และ weights
├── test_pic/          # Test images
├── train.py           # Training script
├── detect.py          # Inference script
├── val.py             # Validation script
├── aug.py             # Data augmentation
├── plot_curve.py      # Plot training curves
└── yolov5s.pt         # Pretrained weights
```

---

## ⚙️ Installation

```bash
git clone https://github.com/thanakritmos12/YOLO_MFK.git
cd YOLO_MFK
pip install -r requirements.txt
```

---

## 🚀 Usage

**Training**
```bash
python train.py --data data/mfk.yaml --weights yolov5s.pt --epochs 100
```

**Detection**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source test_pic/
```

**Validation**
```bash
python val.py --weights runs/train/exp/weights/best.pt --data data/mfk.yaml
```

---

## 🛠️ Tech Stack

- **Model:** YOLOv5
- **Framework:** PyTorch
- **Task:** Object Detection + Segmentation
- **Language:** Python

---
