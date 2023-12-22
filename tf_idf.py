from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import sys
sys.setrecursionlimit(10000)

corpus = ['data science is one of the most important fields of science',
          'this is one of the best data science courses',
          'data scientists analyze data' ]

words_set = set()

for doc in corpus:
    words = doc.split(' ')
    words_set = words_set.union(set(words))

print('Number of words in the corpus:', len(words_set))
print('The words in the corpus: \n', words_set)

n_docs = len(corpus)  # ·Number of documents in the corpus
n_words_set = len(words_set)  # ·Number of unique words in the

df_tf = pd.DataFrame(np.zeros((n_docs, n_words_set)), columns=list(words_set))

# Compute Term Frequency (TF)
for i in range(n_docs):
    words = corpus[i].split(' ')  # Words in the document
    for w in words:
        df_tf[w][i] = df_tf[w][i] + (1 / len(words))

df_tf

print("IDF of: ")

idf = {}

for w in words_set:
    k = 0  # number of documents in the corpus that contain this word

    for i in range(n_docs):
        if w in corpus[i].split():
            k += 1

    idf[w] = np.log10(n_docs / k)

    print(f'{w:>15}: {idf[w]:>10}')

df_tf_idf = df_tf.copy()

for w in words_set:
    for i in range(n_docs):
        df_tf_idf[w][i] = df_tf[w][i] * idf[w]

df_tf_idf


tr_idf_model = TfidfVectorizer()
tf_idf_vector = tr_idf_model.fit_transform(corpus)
print(type(tf_idf_vector), tf_idf_vector.shape)

tf_idf_array = tf_idf_vector.toarray()

print(tf_idf_array)

# words_set = tr_idf_model.get_feature_names()

print(words_set)

df_tf_idf = pd.DataFrame(tf_idf_array, columns=list(words_set))

df_tf_idf
