import os
import pyswarms as ps
import numpy as np
import pickle
import torch
from joblib import delayed, Parallel
# from multiprocessing import cpu_count
from ultralytics import YOLO


# print("Number of CPU cores: {}".format(cpu_count()))
# print(type(cpu_count()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Calculate loss
def lose(code):
    global train_list
    global test_list
    global test_results
    # Clear GPU cache
    torch.cuda.empty_cache()
    result_list = []
    for num in range(code.shape[0]):
        c = cost(code, num, train_list)
        result_list.append(c)
    # result_list = np.zeros(code.shape[0])
    # result_list = Parallel(n_jobs=20)(delayed(cost)(code, num) for num in range(code.shape[0]))
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Get the best code from all candidates
    best_code_num = np.argmin(result_list)
    c = cost(code, best_code_num, test_list)
    print("Above is the result on the test set")
    # Append testing results: [code[0], code[1], c]
    test_results.append([code[best_code_num][0], code[best_code_num][1], c])
    return np.array(result_list)


def cost(code, num, file_list):
    global num_list
    global num_true_list
    n_list = num_list
    # Load all jpg images under unlabeldata and get predictions
    path = 'LabelmeData/'
    result_path = 'LabelmeData_result/'
    # Clear all files under result_path
    for file in os.listdir(result_path):
        os.remove(os.path.join(result_path, file))
    # Process files in sorted order
    for file in file_list:
        if file.endswith('.jpg') or file.endswith('.JPG'):
            img_path = os.path.join(path, file)
            r = model.predict(source=img_path, conf=code[num][0], iou=code[num][1], device=device)
            r = r[0]
            # Get number of predicted labels
            labels = r.boxes.cls.cpu().numpy()
            n_list[file.replace('.jpg', '.txt').replace('.JPG', '.txt')] = len(labels)
    # Compute percentage error for each txt file; ignore those with true count < 10
    result = []
    for file in n_list.keys():
        result.append(np.abs(n_list[file]-num_true_list[file])/num_true_list[file])
    out = np.mean(result)
    print(code[num][0])
    print(code[num][1])
    print(out)
    if (out == 0) | (out == np.nan):
        out = 9999
        return out
    return out

global num_true_list
global num_list
global train_list
global test_list
global test_results
test_results = []
# Load YOLO model
model = YOLO("best.pt")
# Get the true label counts
path = 'LabelmeData/'
# Read number of lines in each txt file
file_list = os.listdir(path)
num_true_list = {}
num_list = {}
# Sort file names
file_list.sort()
for file in file_list:
    if file.endswith('.txt'):
        count = len(open(path + file, 'r').readlines())
        num_true_list[file] = count
        num_list[file] = 0
# Load training and test sets
with open('train.txt', 'rb') as f:
    # Read each line
    train_list = f.readlines()
f.close()
for i in range(len(train_list)):
    line = train_list[i]
    # Extract file names from path
    line = line.decode('utf-8').strip()
    line = line.split('/')[-1]
    train_list[i] = line
with open('val.txt', 'rb') as f:
    test_list = f.readlines()
f.close()
for i in range(len(test_list)):
    line = test_list[i]
    line = line.decode('utf-8').strip()
    line = line.split('/')[-1]
    test_list[i] = line
# Set-up hyperparameters
history = dict()
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
constraints = (np.array([0, 0]), np.array([1, 1]))
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=2, options=options, bounds=constraints)
# Perform optimization
best_cost, best_pos = optimizer.optimize(lose, iters=200)
print(best_pos)
history['best_cost'] = best_cost
history['best_pos'] = best_pos
# Obtain the cost history
history['cost_history'] = optimizer.cost_history
# Obtain the position history+
history['pos_history'] = optimizer.pos_history
# Obtain the velocity history
history['velocity_history'] = optimizer.velocity_history
# save history
with open('history_11_21.pkl', 'wb') as f:
    pickle.dump(history, f)
f.close()
with open('best_pos_11_21.pkl', 'wb') as f:
    pickle.dump(best_pos, f)
with open('test_results_11_21.pkl', 'wb') as f:
    pickle.dump(test_results, f)
f.close()
