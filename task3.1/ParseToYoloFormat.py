import json
import os
import shutil
with open('../dataset/anotations/annotations.json', 'r') as f:
    data = json.load(f)
init_path = "processed"
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
                source = os.path.join("../dataset","images", image['path'])
                destination = os.path.join(currImagePath, f"{image['id']}.jpg")
                imagefound = image
                shutil.copy(source, destination)
                break
        if imagefound is None:
            print(f"Image ID {image_id} not found in the dataset.")
            continue
        for annotation in data['annotations']['pieces']:
            if annotation['image_id'] == image_id:
                # write the anotation in coco format
                height = imagefound['height']
                width = imagefound['width']
                # Convert bounding box to yolo format
                #print(imagefound['path'])

                with open(os.path.join(currAnotationPath, f"{image_id}.txt"), 'a') as f:
                    f.write(f"{annotation['category_id']} {(annotation['bbox'][0]+annotation['bbox'][2]/2)/width} {(annotation['bbox'][1]+annotation['bbox'][3]/2)/height} {annotation['bbox'][2]/width} {annotation['bbox'][3]/height}\n")