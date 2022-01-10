#!/usr/bin/env python
# coding: utf-8

#import pipreqs
#pipreqs /C

import pandas as pd
import numpy as np
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import emoji
from nltk.tokenize import word_tokenize
import nltk
#from textblob import TextBlob
nltk.download('punkt')
import requests
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
#pip install python-Levenshtein

import joblib,pickle
from nltk import FreqDist, classify
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import networkx as nx



import csv
from urllib.request import urlopen
url = 'https://danielutz.pythonanywhere.com/alldata'
response = urlopen(url)
cr = pd.read_csv(response)


df= cr.copy()

#fillling the null data with a default message 
df["article_body"].fillna("No content found in this article body", inplace = True)
df['article_body'] = df.article_body.apply(lambda x: str(x))

#data cleaning
def clean_text(text):
    stopwords = nltk.corpus.stopwords.words('English')
    #w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    pattern =  r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    text = re.sub(pattern, '',text)
    text = re.sub(r'\[.*?\]', '', text)
    text = emoji.demojize(text)  
    text = text.lower() 
    #text = text.isalpha() #remove any numerical number
    text = re.sub(r"\d+", "", text)
    text = re.sub("http\S+|www.\S+", "", text)  # removes URL from string
    #text = " ".join([word for word in text if len(word)>1 and word not in stopwords and word not in string.punctuation])
    text = " ".join(re.split("\s+", text, flags=re.UNICODE))
    text = word_tokenize(text) 
    text = [lemmatizer.lemmatize(w) for w in text]
    clean_token = []
    for word in text:
        if(word not in stopwords and word not in string.punctuation and len(word)>1):
            clean_token.append(word)
        #data_tokens_no_stopwords = [nltk.word_tokenize(t) for t in sents_stopwords_rm]
    
    return " ".join(clean_token)
    #return clean_token
#remove stop words and other left things 

df['clean_article'] = df['article_body'].apply(lambda x: clean_text(x))
df['clean_article']


#print(sentiment_dict
df1 = df.copy()

#Prediction sentiments by machine learning model
#load the saved model
sentiment_classifier = joblib.load('sentiment_analysis_ML_model.pkl')

sentiment_predict=sentiment_classifier.predict(df['clean_article'])

df1['sentiments'] = sentiment_predict

#Prediction Naunced Sentiment by machine learning model
naunced_classifier = joblib.load('naunced_model.pkl')

naunced_predict = naunced_classifier.predict(df['clean_article'])

df1['emotion'] = naunced_predict



#Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer
#data conversion into sparse matrix 
tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                   token_pattern='[a-zA-Z0-9]{3,}',
                                   min_df=10,
                                   max_df=100,
                                   max_features=1000
                                   )

tfidf_matrix = tfidf_vectorizer.fit_transform(df1['clean_article'])


#LDA Topic modelling
#set the topic number in parameter as you wish
LDA = LatentDirichletAllocation(n_components=200, random_state=50)
lda_output = LDA.fit_transform(tfidf_matrix)


#Topwords from LDA model
def show_topics(vectorizer, lda_model, n_words):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=tfidf_vectorizer, lda_model=LDA, n_words=5)

df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords
#topic_keywords



#finding the probablity of topics in all documents (LDA Model)
v = lda_output
v = v*100
#len(v)

#Creating a dataframe for our found probabilities for lda model
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
         #print(feature_name)
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()

        if (max_value - min_value != 0):
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        else:
            result[feature_name] = 0
    return result

LDA_df = pd.DataFrame(v,columns=df_topic_keywords.T.columns)
#NMF_df = pd.DataFrame(d,columns=df_topic_keywords1.T.columns)

LDA_normalized = normalize(LDA_df)

dominant_topic = np.argmax(LDA_normalized.values, axis=1)

#LDANMF['confidence'] = LDANMF.apply (lambda row: computeConfidence(row), axis=1)
LDA_normalized['dominant_topic'] = dominant_topic
LDA_normalized.head()

#concatinating the df1(main file) with LDA and as final
final = pd.concat([df1,LDA_normalized['dominant_topic']],axis=1)
#final.head()

topic_list= topic_keywords

# Converting dominant topics numbers to names
def label_theme(row):
    #row = row['clean_article']
    if (row['dominant_topic'] > len(topic_list) or row['dominant_topic'] < 0):
        return ""
    return topic_list[int(row['dominant_topic'])]

final['dominant_topic_theme'] = final.apply (lambda row: label_theme(row), axis=1)
final.head(10)


final1= final.drop('dominant_topic',axis=1)
#final1.to_csv('Final_data.csv', index=False)



#Extractive Text Summarization

# Read the text and tokenize into sentences
def read_article(text):
    
    sentences =[]
    
    sentences = sent_tokenize(text)
    remove = string.punctuation
    remove = re.sub(r"[.:]+", "", remove)
    for sentence in sentences:
        sentence.replace("[^a-zA-Z0-9]"," ")
        re.sub("http\S+|www.\S+", "", sentence)
        pattern = r"[{}]".format(remove + ':') 
        re.sub(pattern, "", sentence) 
        
    return sentences

# Create vectors and calculate cosine similarity b/w two sentences
def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    #build the vector for the first sentence
    for w in sent1:
        if not w in stopwords:
            vector1[all_words.index(w)]+=1
    
    #build the vector for the second sentence
    for w in sent2:
        if not w in stopwords:
            vector2[all_words.index(w)]+=1
            
    return 1-cosine_distance(vector1,vector2)

# Create similarity matrix among all sentences
def build_similarity_matrix(sentences,stop_words):
    #create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            #print(idx2)
            if idx1!=idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
                
    return similarity_matrix

# Generate and return text summary
def generate_summary(text):
    
    stop_words = stopwords.words('english')
    summarize_text = []
    
    # Step1: read text and tokenize
    sentences = read_article(text)
    
    # Steo2: generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words)
    
    # Step3: Rank sentences in similarirty matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    #Step4: sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    #print(ranked_sentences)
    #l = [x for x in l if x != 0]
    #for i,j in ran
    # Step 5: get the top n number of sentences based on rank    
    summarize_text.append(ranked_sentences[0][1])
    print(summarize_text)
 
    # Step 6 : outpur the summarized version
    return " ".join(summarize_text)

gfinal = final1.copy() 

gfinal['summary']= gfinal['article_body'].apply(generate_summary)

gfinal.to_csv('Final_Data.csv', index=False)

