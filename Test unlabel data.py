# -*- coding: utf-8 -*-
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



unLabelEmbedding = pd.read_csv("dataset/unlabeled_embedding.csv")
train = pd.read_csv("dataset/train_embedding.csv")
devEmbedding = pd.read_csv("dataset/dev_embedding.csv")
 
trainLabel = train.iloc[:,1]
trainData = train.iloc[:,26:-1]

testLabel = devEmbedding.iloc[:,1]
testData = devEmbedding.iloc[:,26:-1]

unLabelData = unLabelEmbedding.iloc[:,1:-1]

ss = StandardScaler()
trainData = ss.fit_transform(trainData)
testData = ss.transform(testData)
unLabelData = ss.transform(unLabelData)

lr2 = LogisticRegression(multi_class='multinomial',solver='newton-cg')
lr2.fit(trainData, trainLabel)
predictionsDev = lr2.predict(testData)
report = classification_report(testLabel, predictionsDev)
print(report)


predictionsUnLabel = lr2.predict(unLabelData)
trainData = pd.DataFrame(trainData)
unLabelData = pd.DataFrame(unLabelData)
predictionsUnLabel = pd.DataFrame(predictionsUnLabel)
trainNewData = pd.concat([trainData, unLabelData], ignore_index = True)
trainNewLabel = pd.concat([trainLabel, predictionsUnLabel], ignore_index = True)

lr3 = LogisticRegression(multi_class='multinomial',solver='newton-cg')
lr3.fit(trainNewData, trainNewLabel)
predictionsDev2  = lr3.predict(testData)
report = classification_report(testLabel, predictionsDev2)
print(report)

