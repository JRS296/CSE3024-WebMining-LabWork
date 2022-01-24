# LAB FAT - CSE3024 (Web Mining)
# Name: Jonathan Rufus Samuel (20BCT0332)
# Slot: L41-L42
# Date & Time: 5:40 pm - 7:20 pm (7.12.2021)
# Question - 1

from re import X
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorize(tokens):
    #It puts 0 if the word is not present in tokens and count of token if present.
    vector=[]
    for w in filtered_vocab:
        vector.append(tokens.count(w))
    return vector
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

stopwords = set(stopwords.words('english'))
special_char = [",",":"," ",";",".","?"]

documentA = "web mining is an interesting subjects, I learned many machine learning concepts"
documentB = "this subject includes many data mining, machine learning and AI concepts"

documentA = documentA.lower()
documentB = documentB.lower()

tokens1 = documentA.split()
tokens2 = documentB.split()
print("\nTokens of Document A: ", tokens1)
print("Tokens of Document B: ",tokens2)
#create a vocabulary list
vocab = unique(tokens1+tokens2) #Makes Sure it is UNIQUE
print("\nFinal Vocabulary of both Documents together: ", vocab)
#filter the vocabulary list
filtered_vocab=[]
for w in vocab: 
    if w not in stopwords and w not in special_char: 
        filtered_vocab.append(w)
print("\nRemoval of Stopwords from both Documents: ",filtered_vocab)
 
CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english') #BoW Algorithm using CountVEctorizer
Count_data = CountVec.fit_transform([documentA,documentB])
 
dataframe = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())
print("\nFinal BoW Summary with Term Frequency & Document Frequency Readings: \n",dataframe)

vector1=vectorize(tokens1)
vector2=vectorize(tokens2)
print("\nDocument Frequency Readings for Document A: \n",vector1)
print("\nDocument Frequency Readings for Document A: \n",vector2)
