# -*- coding: utf-8 -*-
import pandas as pd
import warnings

from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

train = pd.read_csv("dataset/train_embedding.csv")
devEmbedding = pd.read_csv("dataset/dev_embedding.csv")

trainLabel = train.iloc[:,1]
trainData = train.iloc[:,26:-1]

testLabel = devEmbedding.iloc[:,1]
testData = devEmbedding.iloc[:,26:-1]

bernoulliNB = BernoulliNB(alpha = 0.1)
bernoulliNB.fit(trainData, trainLabel)
predictionsNB = bernoulliNB.predict(testData)
report = classification_report(testLabel, predictionsNB)
print("NB Classifier")
print(report)

ss = StandardScaler()
trainData = ss.fit_transform(trainData)
testData = ss.transform(testData)
lr = LogisticRegression(multi_class='multinomial',solver='newton-cg')
lr.fit(trainData, trainLabel)
predictionsLr = lr.predict(testData)
report = classification_report(testLabel, predictionsLr)
print("LR Classifier")
print(report)

ppn = Perceptron(max_iter=500)
ppn.fit(trainData, trainLabel)
predictionsPpn = ppn.predict(testData)
report = classification_report(testLabel, predictionsPpn)
print("Perceptron")
print(report)