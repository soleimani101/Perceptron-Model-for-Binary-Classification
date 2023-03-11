import math
import csv
import numpy as np
import random
import pprint

def sigmoid(z):
    return 1 / (1 + math.exp(-z))
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
def train_perceptron(data, bias, num_iter):
    weights = [0.0] * len(data[0])
    for i in range(num_iter):
        for row in data[1:]:
            predicted_output = predict_output(row, weights, bias)
            error = float(row[-1])- float(predicted_output)
            for j in range(len(row)-1):
                weights[j] += float(error) * float(row[j])
            weights[-1] += float(error) * float(bias)
    return weights
def predict_output(row, weights, bias):
    activation = bias
    for i in range(len(row)-1):
        activation += float(weights[i]) * float(row[i])

    return 1.0 if activation >= 0.0 else 0.0
dataset = load_csv("moons.csv")
dataset = dataset[1:]
random.shuffle(dataset)
train_data = dataset[0:int(len(dataset)*0.8)]
test_data = dataset[int(len(dataset)*0.8):]
weights = train_perceptron(dataset, 0.1, 100)
accuracy = 0.0
for row in test_data:
    predicted_output = predict_output(row, weights, 1)
    if float(predicted_output) == float(row[-1]):
        accuracy += 1
accuracy /= len(test_data)

print(f"Accuracy on test data: {accuracy}")
