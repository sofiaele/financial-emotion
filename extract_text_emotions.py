import pandas as pd
import csv
from skllm.models.gpt import FewShotGPTClassifier
from sklearn.metrics import confusion_matrix, f1_score
from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("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")

# read financial_data_sentiment.csv
df = pd.read_csv('financial_data_sentiment.csv', encoding = "ISO-8859-1")

X = df['text']
y = df['class']

# randomly split into train and test:
X_train = X.sample(frac=0.8, random_state=200)
X_test = X.drop(X_train.index)
y_train = y.sample(frac=0.8, random_state=200)
y_test = y.drop(y_train.index)

# randomly select 100 samples from X_test:
n_test = 100
X_test = X_test.sample(n=n_test, random_state=200)
y_test = y_test.sample(n=n_test, random_state=200)

num_of_train = [5, 10, 15, 20, 25]

f1_s = []
for n_train in num_of_train:
    # randomly select 10 samples from X_train:
    X_train_temp = X_train.sample(n=n_train, random_state=200)
    y_train_temp = y_train.sample(n=n_train, random_state=200)

    print(X_train_temp.shape)
    print(X_test.shape)

    clf = FewShotGPTClassifier()
    clf.fit(X_train_temp, y_train_temp)

    labels = clf.predict(X_test)

    # compute confusion matrix between labels and y_test:
    cm = confusion_matrix(y_test, labels)
    f1 = f1_score(y_test, labels, average='macro')
    print(cm)
    print(f1)
    f1_s.append(f1)

for n, f1 in zip(num_of_train, f1_s):
    print(n, f1)