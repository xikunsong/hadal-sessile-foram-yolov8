import os
import pickle
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO("best.pt")
length_list = {}
# Read test_results.pkl and find the best cost and its corresponding parameters
with open('test_results_11_20.pkl', 'rb') as f:
    test_results = pickle.load(f)
best_result = min(test_results, key=lambda x: x[2])
code = [[best_result[0], best_result[1]]]
num = 0
# Save the cost iteration process to CSV.
# If a later cost is higher than the previous minimum, keep the previous minimum value.
with open('cost_iteration.csv', 'w', newline='') as csvfile:
    fieldnames = ['iteration', 'cost']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, record in enumerate(test_results):
        if i == 0:
            min_cost = record[2]
        else:
            if record[2] < min_cost:
                min_cost = record[2]
        writer.writerow({'iteration': i+1, 'cost': min_cost})
csvfile.close()
# Read history.pkl and save the iteration cost history to CSV
with open('history_11_20.pkl', 'rb') as f:
    history = pickle.load(f)['cost_history']
f.close()
with open('history_iteration.csv', 'w', newline='') as csvfile:
    fieldnames = ['iteration', 'cost']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, record in enumerate(history):
        writer.writerow({'iteration': i+1, 'cost': record})
csvfile.close()
# Read all jpg/JPG files under unlabeldata, perform prediction, and calculate object lengths
path = 'unlabeldata/'
result_path = 'unlabeldata_result/'
# Clear all files in result_path
for file in os.listdir(result_path):
    os.remove(os.path.join(result_path, file))
# Sort file names
file_list = os.listdir(path)
for file in file_list:
    if file.endswith('.jpg') or file.endswith('.JPG'):
        length_list[file] = []
        img_path = os.path.join(path, file)
        r = model.predict(source=img_path, conf=code[num][0], iou=code[num][1], device=device)
        # Save results as txt
        # r[0].save(result_path+file, conf=False, labels=False)
        r[0].save_txt(result_path+file.replace('.jpg', '.txt').replace('.JPG', '.txt'))
        img = cv2.imread(img_path)
        for box in r[0].boxes:
            X1, Y1, X2, Y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
        cv2.imwrite(result_path+file, img)
        # Calculate size of each individual (bounding box diagonal length)
        lengths = []
        for box in r[0].boxes:
            X1, Y1, X2, Y2 = map(int, box.xyxy[0].tolist())
            length = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)
            # Image width corresponds to 10 cm, convert pixel length to actual length (cm)
            img_width = img.shape[1]
            actual_length = length * 10 / img_width
            lengths.append(actual_length)
        if lengths:
            length_list[file].append(lengths)
        else:
            length_list[file].append(0)
# Also calculate lengths for training dataset
train_path = 'LabelmeData/'
train_file_list = os.listdir(train_path)
for file in train_file_list:
    if file.endswith('.jpg') or file.endswith('.JPG'):
        # The txt file corresponding to the image
        txt_file = file.replace('.jpg', '.txt').replace('.JPG', '.txt')
        length_list[file] = []
        # Calculate size for each individual (bounding box diagonal length)
        img = cv2.imread(train_path + file)
        lengths = []
        for line in open(train_path + txt_file, 'r').readlines():
            parts = line.strip().split()
            X1 = int(float(parts[1]) * img.shape[1])
            Y1 = int(float(parts[2]) * img.shape[0])
            X2 = int(float(parts[3]) * img.shape[1])
            Y2 = int(float(parts[4]) * img.shape[0])
            length = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)
             # Image width corresponds to 10 cm, convert pixel length to actual length (cm)
            img_width = img.shape[1]
            actual_length = length * 10 / img_width
            lengths.append(actual_length)
        if lengths:
            length_list[file].append(lengths)
        else:
            length_list[file].append(0)
# Save results to CSV
# File name format: trench_voyage_depth_area.jpg or .JPG
# Record the length of each individual and compute average values
with open('unlabeldata_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Trench', 'Voyage', 'Depth', 'SampleArea', 'IndividualID', 'Length(cm)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for file in length_list.keys():
        parts = file.replace('.jpg', '').replace('.JPG', '').split('_')
        trench = parts[0]
        voyage = parts[1]
        depth = parts[2]
        sample_area = parts[3]
        lengths = length_list[file][0]
        if lengths == 0:
            continue
        for i, length in enumerate(lengths):
            writer.writerow({'Trench': trench, 'Voyage': voyage, 'Depth': depth, 'SampleArea': sample_area, 'IndividualID': i+1, 'Length(cm)': length})
csvfile.close()