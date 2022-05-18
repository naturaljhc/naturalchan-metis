#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np
import datetime as dt
import spacy
import string
import time
import demoji
import re
import cloudpickle as cpkl
import pickle as pkl
import scipy
import shutil
from sqlalchemy import create_engine
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import AgglomerativeClustering
from nltk.chunk import ne_chunk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import matplotlib.pyplot as plt

os.chdir('/home/chanjunho/reddit_context_bot')

# Load spacy english language
nlp = spacy.load('en_core_web_sm')


# In[2]:


def topwords(topicstoterms):
    """
    topicstoterms - mxn matrix where m is the number of topics we searched for
                    and n is the number of tokens in the dtm
    This function will report the most important words in each topic in the
    form of a DataFrame
    """
    topictop10 = pd.DataFrame()
    transposed = topicstoterms.transpose()
    for i in range(transposed.shape[1]):
        topictop10[i] = transposed[i].sort_values(ascending = False)[0:10].index
    return topictop10

def applyNMF(dtm, vectorizer, n_topics):
    """
    dtm - Document-Term Matrix (Use Sparse matrix for efficiencity)
    vectorizer - The vectorizer (CountVectorizer, TFIDF) we would like to use
    n_topics - Number of topics we want to search for
    This function will create a fit a NMF model on the dtm and then return the model
    along with a DataFrame on the topics to terms
    """
    nmf = NMF(n_topics, init = "nndsvd")
    nmf.fit(dtm)
    return nmf, pd.DataFrame(nmf.components_, columns = vectorizer.get_feature_names_out())

def topic_plot(topics_dtm, topics, filename):
    """
    topics_dtm - A sparse matrix describing each document and it's relevance in each topic
    topics - Name of each topic (determined manually)
    filename - File to save image to
    This function plots the number of documents belonging in each topic
    """
    topic_count = []
    for col in topics_dtm.columns:
        topic_count.append(sum(topics_dtm[col].astype(bool)))
        topic_dict = dict(zip(topics.values(), topic_count))
    topic_dict = dict(sorted(topic_dict.items(), key=lambda item: item[1], reverse = True))
    plt.bar(topic_dict.keys(), topic_dict.values())
    plt.title("Distribution of Topics", fontsize = 18)
    plt.xlabel("Topics", fontsize = 18)
    plt.ylabel("Topic Count", fontsize = 18)
    plt.xticks(rotation = 90)
    plt.savefig(filename, bbox_inches = "tight")
    plt.show()

def spacy_tokenizer(sentence):
    """
    sentence - A string (essentially a document)
    This function is our custom tokenizer in order to clean our documents/posts
    efficiently. It will remove quotation marks, non-nouns/non-verbs, and lemmatize
    any appropriate words
    """
    all_tokens = nlp(sentence)
    
    tokens = [token.lemma_.lower().strip() for token in all_tokens if (token.pos_ in {"NOUN", "PROPN", "VERB"} and token.is_quote == False and len(token) > 1)]
    lemmatized = [word.strip("\"'") for word in tokens if word not in stopwords]
    return lemmatized

def remove_emoji(sentence):
    """
    sentence - A string (essentially a document)
    This function removes emojis (may be redundant when remove all non-ascii characters)
    """
    emojis = demoji.findall(sentence)
    for emoji in emojis.keys():
        sentence = sentence.replace(emoji, '')
    return sentence

def cosine_sim(a, b):
    """
    a, b - Vectors
    This is a custom cosine similarity function
    """
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1
    else:
        return 1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


# In[5]:


engine = create_engine('sqlite:///models/posts.db')
conn = engine.connect()

df = pd.read_sql('SELECT * FROM posts', 
                 con = engine)

conn.close()
# # Some cleaning due to weird fonts on the letters T and B
# df["title"].update(pd.Series(["Canada will list the Proud Boys movement as a terrorist group",
#                          "Scottish Parliament to hold vote on Unexplained Wealth Order into Donald Trump's finances | MSPs will be asked to vote this week on whether the Scottish Government should pursue an Unexplained Wealth Order (UWO) to investigate the source of financing for Donald Trump’s Scottish resorts.",
#                          "With reports that Donald Trump may fly to Scotland after he leaves office, BrewDog have started a petition to rename Prestwick Airport ‘Joe Biden International’.", 
#                          "No, Trump won’t be golfing in Scotland on Inauguration Day, First Minister Nicola Sturgeon says | Pandemic travel ban applies to President Trump as much as anybody else, the Scotland’s head of government says",
#                          "Russian Media Wants Moscow to Grant Asylum to Trump",
#                          "Bolsonaro abandons 'friend' Trump after 2020 election, says he's 'not the most important person in the world'"],
#                          index = [9296, 9331, 9602, 9892, 10425, 11068]))

# Some cleaning and creating a pandas Series of all the cleaned documents
corpus = []
for sentence in df["title"]:
    sentence = remove_emoji(sentence)
    for c in sentence:
        if ord(c) >= 128:
            sentence = sentence.replace(c, ' ')
    corpus.append(sentence.replace('\n', '').replace('\t', '').replace('\r', '').replace('  ', ' '))
corpus = pd.Series(corpus)


# In[6]:


stopwords = nlp.Defaults.stop_words


# ***

# ***

# ***

# # CountVectorizer w/ Custom Stop Words & SpaCy Lemmatization

# In[7]:


cv = CountVectorizer(tokenizer = spacy_tokenizer)
X_cv = cv.fit_transform(corpus).toarray()
dtm_spacycv = pd.DataFrame(X_cv, columns = cv.get_feature_names_out())
sps_spacycv = scipy.sparse.csr_matrix(dtm_spacycv)


# In[8]:


nmfcv10, nmfcv10_topicstoterms = applyNMF(sps_spacycv, cv, 20)


# In[9]:


nmfcv10_topics_dtm = pd.DataFrame(nmfcv10.transform(sps_spacycv))


# In[10]:


with open("models/cv.pkl", "wb") as file_name:
    cpkl.dump(cv, file_name)
with open("models/nmf.pkl", "wb") as file_name:
    cpkl.dump(nmfcv10, file_name)
conn = engine.connect()
corpus.to_sql('corpus', con = engine, if_exists = 'replace', index = False)
nmfcv10_topics_dtm.to_sql('dtm', con = engine, if_exists = 'replace', index = False)
conn.close()
engine.dispose()
# shutil.copyfile('posts.db', './context_app/models/posts.db')


# # Testing

# In[246]:


# def recommend_posts(new_post, vectorizer, model, train_dtm):
#     X_test = vectorizer.transform([new_post]).toarray()
#     dtm_test = pd.DataFrame(X_test, columns = vectorizer.get_feature_names_out())
#     sps_test = scipy.sparse.csr_matrix(dtm_test)
#     post = np.array(pd.DataFrame(model.transform(sps_test))).flatten()
#     print(post)
#     distances = []
#     for i in range(train_dtm.shape[0]):
#         distances.append(cosine_sim(post, train_dtm.iloc[i,:]))
#     distances = pd.Series(distances).sort_values()
#     return pd.DataFrame(np.array([corpus[distances[:10].index], distances[:10]]).T, columns = ["title", "cosine_similarity"], index = distances[:10].index)


# In[314]:


# corpus_test[616]


# In[313]:


# recommend_posts(corpus_test[616], cv, nmfcv10, nmfcv10_topics_dtm)


# In[238]:


# recommend_posts(corpus_test[616], tfidf, nmftfidf10, nmftfidf10_topics_dtm)


# In[ ]:




