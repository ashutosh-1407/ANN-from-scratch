# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:58:45 2020

@author: ashut
"""

import numpy as np
#import matplotlib.pyplot as plt
import time, sys, datetime
import csv

def load_train_test_data(train_image_file, train_label_file, test_image_file):
    
    #Loading train_image.csv

    rows = []
    counter = 0
    csv_data_file = open(train_image_file, 'r')
    csv_data = csv.reader(csv_data_file)
    for row in csv_data:
        rows.append(row)
    X_train = np.zeros((len(rows), 784))
    for i in range(len(rows)):
        X_train[counter] = rows[counter]
        counter+=1
     
    #Loading train_label.csv
        
    rows = []
    counter = 0
    csv_data_file = open(train_label_file, 'r')
    csv_data = csv.reader(csv_data_file)
    for row in csv_data:
        rows.append(row)
    y_train_int = np.zeros((len(rows)))
    for i in range(len(rows)):
        y_train_int[counter] = rows[counter][0]
        counter+=1
        
    #Loading test_image.csv

    rows = []
    counter = 0
    csv_data_file = open(test_image_file, 'r')
    csv_data = csv.reader(csv_data_file)
    for row in csv_data:
        rows.append(row)
    X_test = np.zeros((len(rows), 784))
    for i in range(len(rows)):
        X_test[counter] = rows[counter]
        counter+=1

    #Image preprocessing.
    #X_train = X_train[0:10000]
    #y_train_int = y_train_int[0:10000]
    X_train = X_train.astype('int64')
    y_train_int = y_train_int.astype('int64')
    X_test = X_test.astype('int64')

    X_train = np.asfarray(X_train) * (0.99 / 255) + 0.01
    X_test = np.asfarray(X_test) * (0.99 / 255) + 0.01

    y_train = np.eye(np.max(y_train_int) + 1)[y_train_int]

    y_train[y_train==0] = 0.01
    y_train[y_train==1] = 0.99
    
    return X_train, y_train, y_train_int, X_test


#Building the model.

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)

def init_model(input_dim, no_of_hidden_layers, output_dim):
    model = {}
    hidden_dim = 128
    model['W1'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    model['b1'] = np.zeros((1, hidden_dim))
    for i in range(no_of_hidden_layers - 1):
        weight = 'W' + str(i + 2)
        bias = 'b' + str(i + 2)
        model[weight] = np.random.randn(hidden_dim, hidden_dim // 2)
        model[bias] = np.zeros((1, hidden_dim // 2))
        hidden_dim//=2
    model['W' + str(no_of_hidden_layers + 1)] = np.random.randn(hidden_dim, output_dim)
    model['b' + str(no_of_hidden_layers + 1)] = np.zeros((1, output_dim))
    return model
        
def retrieve(model_dict):
    weight_bias = []
    for i in range(len(model_dict.keys()) // 2):
        weight_bias.append(model_dict['W' + str(i + 1)])
        weight_bias.append(model_dict['b' + str(i + 1)])
    return weight_bias

def feedforward(X, model_dict):
    weight_bias = retrieve(model_dict)
    activations = []
    z = X.dot(weight_bias[0]) + weight_bias[1]
    for i in range(2, len(model_dict.keys()), 2):
        a = sigmoid(z)
        activations.append(a)
        z = a.dot(weight_bias[i]) + weight_bias[i + 1]
    activations.append(softmax(z))
    return activations
      
def backpropogate(X, y, model_dict, activations):
    weight_bias = retrieve(model_dict)
    
    idx = len(activations)
    derv_wrt_a = y - activations[idx - 1]
    updated_weight_bias = []
    for i in range(len(activations) - 1):
        derv_a = derv_wrt_a * activations[idx - 1] * (1 - activations[idx - 1])
        updated_weight_bias.insert(0, np.dot(activations[idx - 2].T, derv_a))
        updated_weight_bias.insert(1, np.sum(derv_a, axis = 0, keepdims = True))
        derv_wrt_a = np.dot(derv_wrt_a, weight_bias[2 * (idx - 1)].T)
        idx-=1
    
    derv_a = derv_wrt_a * activations[idx - 1] * (1 - activations[idx - 1])
    updated_weight_bias.insert(0, np.dot(X.T, derv_a))
    updated_weight_bias.insert(1, np.sum(derv_a, axis = 0, keepdims = True))
    
    new_idx = 0
    for i in range(len(weight_bias) // 2):
        updated_weight = weight_bias[new_idx] + (learning_rate * updated_weight_bias[new_idx])
        updated_bias = weight_bias[new_idx + 1] + (learning_rate * updated_weight_bias[new_idx + 1])
        model_dict['W' + str(i + 1)] = updated_weight
        model_dict['b' + str(i + 1)] = updated_bias
        new_idx+=2
        
def train(X, y, model_dict):
    global current_image, batch_size
    X = np.array(X, ndmin = 2)
    y = np.array(y, ndmin = 2)
    activations = feedforward(X, model_dict)
    backpropogate(X, y, model_dict, activations)
    
def predict_label(X, model_dict):
    X = np.array(X, ndmin = 2)
    activations = feedforward(X, model_dict)
    return activations[len(activations) - 1]
    
def evaluate_model(X, y, model_dict): 
    global loss
    correct_preds, wrong_preds = 0, 0
    for i in range(len(X)):
        y_i_preds = predict_label(X[i], model_dict)
        y_i = y_i_preds.argmax()
        loss.append(y[i] - y_i)
        if y_i == y[i]:
            correct_preds+=1
        else:
            wrong_preds+=1
    return correct_preds / (correct_preds + wrong_preds)
  
    
input_neurons = 784
output_neurons = 10
learning_rate = 0.001
start_time = time.time()
print(datetime.datetime.now())

if len(sys.argv) == 1:
    train_image_file = 'train_image.csv'
    train_label_file = 'train_label.csv'
    test_image_file = 'test_image.csv'
elif len(sys.argv) == 2:
    train_image_file = sys.argv[1]
    train_label_file = 'train_label.csv'
    test_image_file = 'test_image.csv'
elif len(sys.argv) == 3:
    train_image_file = sys.argv[1]
    train_label_file = sys.argv[2]
    test_image_file = 'test_image.csv'
elif len(sys.argv) >= 4:
    train_image_file = sys.argv[1]
    train_label_file = sys.argv[2]
    test_image_file = sys.argv[3]   
   
X_train, y_train, y_train_int, X_test = load_train_test_data(train_image_file, train_label_file, test_image_file)

if X_train.shape[0] >= 0 and X_train.shape[0] < 20000:
	epochs = 100
elif X_train.shape[0] >= 20000 and X_train.shape[0] < 40000:
	epochs = 20
elif X_train.shape[0] >= 40000:
	epochs = 10

model_dict = init_model(input_dim = input_neurons, no_of_hidden_layers = 2, output_dim = output_neurons)
model_loss = []
model_accuracy = []

print('\nX_train shape', X_train.shape)
print('y_train_int shape', y_train_int.shape)
print()

for i in range(epochs):
    loss = []
    print('Training for epoch {}'.format(i + 1))
    for img_index in range(X_train.shape[0]):
        counter = img_index + 1
#        str_counter = 'Number of images processed: ' + str(counter)
#        sys.stdout.write('\r'+str_counter)
        train(X_train[img_index], y_train[img_index], model_dict)
#    train_accuracy = evaluate_model(X_train, y_train_int, model_dict)
#    print()
#    print("="*50)
#    print('Accuracy on train images is {:.3f}'.format(train_accuracy))
#    test_accuracy = evaluate_model(X_test, y_test_int, model_dict)
#    print('Accuracy on test images is {:.3f}'.format(test_accuracy))
#    print("="*50)
#    print()
#    model_accuracy.append(test_accuracy)
#    l = np.sum(np.square(loss)) / len(loss)
#    model_loss.append(l)


#plt.plot(model_loss)
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.title('Loss Graph')
#plt.show()
#plt.plot(model_accuracy)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.title('Accuracy Graph')
#plt.show()

#Predicting and creating the output file.

output_file = open('test_predictions.csv', 'w')

for i in range(len(X_test)):
    y_pred_probs = predict_label(X_test[i], model_dict)
    y_pred = y_pred_probs.argmax()
    output_file.write(str(y_pred))
    output_file.write('\n')
    
#rows = []
#counter = 0
#csv_data_file = open('test_label.csv', 'r')
#csv_data = csv.reader(csv_data_file)
#for row in csv_data:
#	rows.append(row)
#y_int = np.zeros((len(rows)))
#for i in range(len(rows)):
#	y_int[counter] = rows[counter][0]
#	counter+=1
#    
#y_int = y_int.astype('int64')
#    
#y_preds = np.eye(np.max(y_int) + 1)[y_int]
#y_preds[y_preds==0] = 0.01
#y_preds[y_preds==1] = 0.99

output_file.close()

#print("Report is as follows:")
#test_accuracy = evaluate_model(X_test, y_int, model_dict)
#print()
#print("="*50)
#print('Accuracy on test images is {:.3f}'.format(test_accuracy))
#print("="*50)
#print()

print(datetime.datetime.now())

end_time = time.time()
print('Total time elapsed is {:.3f} minutes'.format((end_time - start_time) / 60))
