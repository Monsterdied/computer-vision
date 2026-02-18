# Chessboard Digitization & Digital Twin Pipeline

This project implements a complete end-to-end Computer Vision pipeline to convert real-world photographs of chessboards into accurate digital representations (Digital Twins). The system achieves a **95.5% total board accuracy** on the ChessReD dataset.

## 🚀 Key Features

- **Piece Counting (Task 2):** Developed a regression-based CNN using **ResNeXt101** with transfer learning. By utilizing a scaled Sigmoid activation, the model achieves a **Mean Absolute Error (MAE) of 0.671**.
- **Object Detection (Task 3.1):** Fine-tuned **YOLOv11** models to detect and classify 12 distinct chess piece types with high precision (**mAP50-95 of 0.885**).
- **Digital Twin Generation (Task 3.2):** Engineered a spatial mapping logic that combines YOLO pose detection (to find board corners) with IOU-based piece placement to reconstruct the 64-square grid.

## 📊 Visual Results

The pipeline successfully handles perspective distortion through a warping process and maps 2D detections to a digital grid.

| Ground Truth | Warped Perspective | Predicted Digital Board |
|:---:|:---:|:---:|
| *Original Board State* | *Geometric Rectification* | *AI Reconstruction (Digital Twin)* |

*(Note: Ensure the image filenames above match the files in your ./Docs/ folder, e.g., Figure3.png, Figure4.png, etc.)*

## 🛠️ Technical Stack
- **Frameworks:** PyTorch, Ultralytics (YOLOv11)
- **Architectures:** ResNeXt101, ResNet50, YOLOv11 (n/s/m/l/x)
- **Libraries:** OpenCV, NumPy, Matplotlib
- **Techniques:** Transfer Learning, Image Warping (Homography), Multi-label Classification, Regression.

## 📂 Documentation & Notebooks

- **Detailed Project Report:** [Read the full PDF analysis here](./Docs/computer_vision.pdf)
- **Final Implementation:** [Task 3.2 Final Notebook](./deliveries/task3_2_Tom%C3%A1s.ipynb)

---
*Developed as part of the Computer Vision curriculum at FEUP.*
# Usage
create an input file with the name `input.json` in same directoy as this script.
```json
{
  "image_files": [
    "images/G000_IMG062.jpg",
    "images/G000_IMG087.jpg"
  ]
}
```
Next execute the script with the following command:

```Bash
python ./script.py
```

The result should be saved in the same directory as `output.json`
