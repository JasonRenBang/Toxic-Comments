# -*- coding: utf-8 -*-
import pandas as pd
import warnings
import numpy as np 

from sklearn.metrics import classification_report


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))    
    return s
    

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]      

    A = sigmoid(np.dot(w.T, X) + b)          

    cost = -1 / m *  np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))   
    dw = 1 / m * np.dot(X, (A - Y).T)   
    db = 1 / m * np.sum(A - Y)   
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())    
    grads = {"dw": dw,
             "db": db}    
    return grads, cost



def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []    
    for i in range(num_iterations):    
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]              
        db = grads["db"]
        w = w - learning_rate * dw   
        b = b - learning_rate * db
        if i % 50 == 0:
            costs.append(cost)
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs



def predict(w, b, X):
	m = X.shape[1]
	Y_prediction = np.zeros((1,m))
	A = sigmoid(np.dot(w.T, X) + b)
	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_prediction[0, i] = 0
		else:
			Y_prediction[0, i] = 1
	assert(Y_prediction.shape == (1, m))
    
	return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    
    w, b = initialize_with_zeros(X_train.shape[0]) 
    
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    predictions = Y_prediction_test.reshape(Y_prediction_test.shape[0],-1).T   

    return predictions




warnings.filterwarnings('ignore')

train = pd.read_csv("dataset/train_embedding.csv")
devEmbedding = pd.read_csv("dataset/dev_embedding.csv")
trainLabel = train.iloc[:,1]
trainData = train.iloc[:,26:-1]
trainLabel = np.array(trainLabel)
trainData = np.array(trainData)
trainData = trainData.reshape(trainData.shape[0],-1).T

testLabel = devEmbedding.iloc[:,1]
testData = devEmbedding.iloc[:,26:-1]
testLabel = np.array(testLabel)
testData = np.array(testData)
testData = testData.reshape(testData.shape[0],-1).T

predictions = model(trainData, trainLabel, testData, testLabel, num_iterations = 500, learning_rate = 0.01)
report = classification_report(testLabel, predictions)
print(report)
