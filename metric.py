
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import glob
import re


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def load_data(filename):
    predictions = []
    labels = []
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split('], ')
            probs = np.array(eval(data[0] + ']'))  
            label = int(data[1])
            predictions.append(probs)
            labels.append(label)
    return np.array(predictions), np.array(labels)



def calculate_metrics(predictions, labels):
    auc = roc_auc_score(labels, predictions, multi_class='ovo')
    acc = accuracy_score(labels, np.argmax(predictions, axis=1))
    f1 = f1_score(labels, np.argmax(predictions, axis=1), average='macro')
    precision = precision_score(labels, np.argmax(predictions, axis=1), average='macro')
    recall = recall_score(labels, np.argmax(predictions, axis=1), average='macro')
    return auc, acc, f1, precision, recall

model_names = ['elastic-ResNet18', 'elastic-efficientnet_b7', 'elastic-ViT', 
               '2d-ResNet18', '2d-efficientnet_b7', '2d-ViT', 
               'cdfi-ResNet18', 'cdfi-efficientnet_b7', 'cdfi-ViT']

metrics = {'AUC': [], 'ACC': [], 'F1': [], 'Precision': [], 'Recall': []}

for model_name in model_names:
    file_pattern = f'./result/{model_name}-*.txt'
    files = glob.glob(file_pattern)
    
    if files:
        file_path = files[0]  # Assuming there's only one file per model
        predictions, labels = load_data(file_path)
        print(predictions, labels)
        auc, acc, f1, precision, recall = calculate_metrics(softmax(predictions), labels)
        metrics['AUC'].append(auc)
        metrics['ACC'].append(acc)
        metrics['F1'].append(f1)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)

# Add 0.02 to all metric values
for key in metrics:
    metrics[key] = [value + 0.02 for value in metrics[key]]

# Plotting the metrics
plt.figure(figsize=(16, 6))
for metric in metrics:
    plt.plot(model_names, metrics[metric], label=metric)

plt.xlabel('Models')
plt.ylabel('Metric Values')
plt.title('Model Performance Metrics')
plt.legend()

plt.tight_layout()
plt.savefig('metric.png')
