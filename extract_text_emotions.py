#
import pandas as pd
import csv

# read financial_data_sentiment.csv

df = pd.read_csv('financial_data_sentiment.csv', encoding = "ISO-8859-1")

from skllm.config import SKLLMConfig
SKLLMConfig.set_openai_key("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")
#SKLLMConfig.set_openai_org("Personal")

X = df['text']
y = df['class']

X_train = X[0:4000:400]
y_train = y[0:4000:400]

X_test = X[4002::20] 
y_test = y[4002::20]

print(X_train.shape)
print(X_test.shape)

from skllm.models.gpt import FewShotGPTClassifier

clf = FewShotGPTClassifier()
clf.fit(X_train, y_train)

labels = clf.predict(X_test)
print(labels)

# compute confusion matrix between labels and y_test:
from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(y_test, labels)
f1 = f1_score(y_test, labels, average='macro')
print(cm)
print(f1)