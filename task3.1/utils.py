import requests
import os
import zipfile
def download_file(url, save_path):
    """Download a file from a URL and save it to the specified path"""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad status codes
    
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    return save_path
def extract_zip(zip_path, extract_dir):
    """Extract all files from a ZIP archive"""
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_dir)
    return extract_dir

def absolute_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    # Calculate width and height of bounding box
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    # Calculate center coordinates
    x_center = x_min + (box_width / 2)
    y_center = y_min + (box_height / 2)
    
    # Normalize coordinates
    x_center /= img_width
    y_center /= img_height
    box_width /= img_width
    box_height /= img_height
    
    return f" {x_center} {y_center} {box_width} {box_height}"