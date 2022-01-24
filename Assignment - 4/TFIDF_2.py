import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# Train & Test Data
train = ['This movie is very scary and long','This movie is not scary and is slow']
test = ['This movie is pretty scary, spooky and good. Overall, a good, long, but slow movie.']

# instantiate the vectorizer object
countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')

# convert th documents into a matrix
count_w = countvectorizer.fit_transform(train)
tfidf_w = tfidfvectorizer.fit_transform(train)

#count_tokens = tfidfvectorizer.get_feature_names() 
count_tokens = countvectorizer.get_feature_names()
tfidf_tokens = tfidfvectorizer.get_feature_names()
df_countvect = pd.DataFrame(data = count_w.toarray(),index = ['Doc1','Doc2'],columns = count_tokens)
df_tfidfvect = pd.DataFrame(data = tfidf_w.toarray(),index = ['Doc1','Doc2'],columns = tfidf_tokens)
print("Count Vectorizer\n")
print(df_countvect)
print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect)