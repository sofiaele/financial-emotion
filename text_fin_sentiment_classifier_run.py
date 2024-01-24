import pandas as pd
import csv
from skllm.models.gpt import FewShotGPTClassifier
from sklearn.metrics import confusion_matrix, f1_score
from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("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")

# read data/FOMC\ Press\ Conference\ April\ 25,\ 2012.csv
df = pd.read_csv('data/FOMC Press Conference April 25, 2012.csv', encoding = "ISO-8859-1")

sentences = df['text']

# use pickle to load: financial_sentiment_classifier.pkl
import pickle
with open('financial_sentiment_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

y = clf.predict(sentences[0::3])
print(y)