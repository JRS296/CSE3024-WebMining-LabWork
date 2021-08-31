#import count vectorize and tfidf vectorise
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
train = ('The sky is blue.','The sun is bright.')
test = ('The sun in the sky is bright', 'We can see the shining sun, the bright sun.')
# instantiate the vectorizer object
# use analyzer is word and stop_words is english which are responsible for remove stop words and create word vocabulary
countvectorizer = CountVectorizer(analyzer='word' , stop_words='english')
terms = countvectorizer.fit_transform(train)
term_vectors  = countvectorizer.transform(test)
print("Sparse Matrix form of test data : \n")
print(term_vectors.todense())

# instantiate the vectorizer object
# use analyzer is word and stop_words is english which are responsible for remove stop words and create word vocabulary

tfidfvectorizer = TfidfVectorizer(analyzer='word' , stop_words='english',)
tfidfvectorizer.fit(train)
tfidf_train = tfidfvectorizer.transform(train)
tfidf_term_vectors  = tfidfvectorizer.transform(test)
print("Sparse Matrix form of test data : \n")
tfidf_term_vectors.todense()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(norm='l2')
term_vectors.todense()
#[0, 1, 1, 1]
# [0, 1, 0, 2]
tfidf.fit(term_vectors)
tf_idf_matrix = tfidf.transform(term_vectors)
print("\nVector of idf \n")
print(tfidf.idf_)
print("\nFinal tf-idf vectorizer matrix form :\n")
print(tf_idf_matrix.todense())