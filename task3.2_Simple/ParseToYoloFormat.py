import json
import os
import shutil
with open('../annotations.json', 'r') as f:
    data = json.load(f)
init_path = "datasets"
if os.path.exists(init_path):
    #os.remove(init_path)
    print("Directory already exists. Please remove it manually.")
chessRed2k = True

if not chessRed2k:
    init_path = os.path.join(init_path, "full")
    images=data['splits']
else:
    init_path = os.path.join(init_path, "2k")
    images=data['splits']['chessred2k']


for split in data['splits']['chessred2k'].keys():
    print(f"Split: {split}")
    currImagePath = os.path.join(init_path, split, "images")
    os.makedirs(currImagePath)
    currAnotationPath = os.path.join(init_path, split,"labels")
    os.makedirs(currAnotationPath)
    for image_id in images[split]['image_ids']:
        #print(f"Image ID: {image_id}")
        # write the image in coco format
        imagefound = None
        for image in data['images']:
            if(image['id'] == image_id):
                source = os.path.join("../", image['path'])
                destination = os.path.join(currImagePath, f"{image['id']}.jpg")
                imagefound = image
                shutil.copy(source, destination)
                break
        if imagefound is None:
            print(f"Image ID {image_id} not found in the dataset.")
            continue
        for annotation in data['annotations']['corners']:
            if annotation['image_id'] == image_id:
                height = imagefound['height']
                width = imagefound['width']
                
                # Extract and normalize keypoints
                keypoints = []
                coners = [annotation["corners"]["top_left"],annotation["corners"]["top_right"],annotation["corners"]["bottom_left"],annotation["corners"]["bottom_right"]]
                for point in coners:
                    x, y = point[0] / width, point[1] / height
                    keypoints.extend([x, y])  # Add visibility=1 for visible points
                
                # Calculate bounding box (YOLO format: center_x, center_y, width, height)
                all_x = [kp[0] for kp in annotation["corners"].values()]
                all_y = [kp[1] for kp in annotation["corners"].values()]
                x_min, x_max = min(all_x)/width, max(all_x)/width
                y_min, y_max = min(all_y)/height, max(all_y)/height
                
                bbox_center_x = (x_min + x_max) / 2
                bbox_center_y = (y_min + y_max) / 2
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                
                # Class ID (0 for "board")
                class_id = 0
                
                # Write to file (YOLOv8 pose format)
                with open(os.path.join(currAnotationPath, f"{image_id}.txt"), 'a') as f:
                    # Format: class bbox_x bbox_y bbox_w bbox_h kp1_x kp1_y kp1_v ... kpN_x kpN_y kpN_v
                    line = f"{class_id} {bbox_center_x:.6f} {bbox_center_y:.6f} {bbox_width:.6f} {bbox_height:.6f} " + \
                        " ".join([f"{kp:.6f}" for kp in keypoints]) + "\n"
                    f.write(line)