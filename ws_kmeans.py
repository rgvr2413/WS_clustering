# Hierarchial clustering on the shakespere plays

import sklearn
import pandas as pd
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import contractions
import re

#nltk.download('stopwords')

stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    doc = contractions.fix(doc)
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    #filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

file_list = glob.glob('D:\Work\Research papers\WS\*.txt')
ws = []

for file in file_list:
    with open(file, 'r') as f:
        ws.append(f.read())

#print(ws)

ws_df = pd.DataFrame(ws)

print(ws_df)

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(ws)

k = 8

km = KMeans(n_clusters=k)
km.fit(vectors)

yp = km.predict(vectors)

#print(km.labels_)
print(yp)

ws_df['kmeans cluster'] = km.labels_

print(ws_df)
x = vectors.toarray()

print(vectors[:, 0])
sns.scatterplot(x=vectors[:, 0], y=vectors[:, 5], hue=yp, palette='rainbow')
plt.show()