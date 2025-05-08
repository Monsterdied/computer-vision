from ultralytics import YOLO
import torch
import ultralytics.nn.tasks

def main():
# Load YOLOv10n model from scratch
    #torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
    model = YOLO('yolo11x.pt')
    # Train the model
    #d:/A_University/ano4/semester_2/Vision/computer-vision/processed/res2k.yaml
    model.train(data="D:/A_University/ano4/semester_2/Vision/computer-vision/processed/res2k.yaml", epochs=20, imgsz=1850,batch=1,workers=6,name = "Yolo11x")


if __name__ == '__main__':
    main()
