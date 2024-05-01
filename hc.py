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
from scipy.cluster.hierarchy import ward, dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity
import os

#nltk.download('stopwords')
#nltk.download('punkt')

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
ws1 = []
file_name = []

for file in file_list:
    head, fl = os.path.split(file)
    file_name.append(fl)
    with open(file, 'r') as f:
        ws.append(f.read())

# Create linkage matrix using cosine similarity
def ward_hierarchical_clustering(feature_matrix): 
    
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix

#print(ws)

for w in ws:
    w = normalize_document(w)
    ws1.append(w)

ws_df = pd.DataFrame(ws1)

#print(ws_df)

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(ws1)     #This is a vectorized matrix

dist_matrix = 1 - cosine_similarity(vectors)    # Compute document similarity or distance
print(dist_matrix)

linkage_matrix = linkage(dist_matrix, method='complete')

#print(linkage_matrix)

plt.figure(figsize=(10, 10))
dendrogram(linkage_matrix, labels=
           file_name, leaf_rotation=90)
plt.show()