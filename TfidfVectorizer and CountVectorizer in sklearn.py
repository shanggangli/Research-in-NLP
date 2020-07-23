 # 1. TfidfVectorizer
 
from sklearn.feature_extraction.text import TfidfVectorizer


text = ["The quick brown fox jumped over the lazy dog."]

vectorizer=TfidfVectorizer()
vectorizer.fit(text)
print(vectorizer.vocabulary_) #{'the': 7, 'quick': 6, 'brown': 0, 'fox': 2, 'jumped': 3, 'over': 5, 'lazy': 4, 'dog': 1}

vector=vectorizer.transform(text)
print(vector.toarray()) #[[0.30151134 0.30151134 0.30151134 0.30151134 0.30151134 0.30151134,0.30151134 0.60302269]]


# 2.CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

text = ["The quick brown fox jumped over the lazy dog."]

vectorizer=CountVectorizer()
vectorizer.fit(text)
bag_of_word=vectorizer.transform(text)
sum_words=bag_of_word.sum(axis=0)
print(sum_words) #[[1 1 1 1 1 1 1 2]]

words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

print(words_freq)   #[('the', 2), ('quick', 1), ('brown', 1), ('fox', 1), ('jumped', 1), ('over', 1), ('lazy', 1), ('dog', 1)]

print(vectorizer.vocabulary_) #{'the': 7, 'quick': 6, 'brown': 0, 'fox': 2, 'jumped': 3, 'over': 5, 'lazy': 4, 'dog': 1}

print(vectorizer.vocabulary_.items()) #dict_items([('the', 7), ('quick', 6), ('brown', 0), ('fox', 2), ('jumped', 3), ('over', 5), ('lazy', 4), ('dog', 1)])

vector=vectorizer.transform(text)
print(vector.toarray())     #[[1 1 1 1 1 1 1 2]]
