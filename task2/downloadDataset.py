from task2.utils import download_file, extract_zip
import os

if not os.path.exists("dataset/"):
    os.makedirs("dataset/")
    print("Downloading chessboard images...")
    download_file("https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/6329e969-616e-48e3-b893-a0379d1c15ba", "chessboard.zip")
    os.makedirs("dataset/images/")
    print("Extracting chessboard images...")
    extract_zip("chessboard.zip", "dataset/images/")
    os.remove("chessboard.zip")
    os.makedirs("dataset/annotations/")
    print("Downloading chessboard annotations...")
    download_file("https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/3cae6364-daca-4967-b426-1e4b68cdb64c", "annotations.zip")
    print("Extracting chessboard annotations...")
    extract_zip("annotations.zip", "dataset/annotations/")
    os.remove("annotations.zip")
