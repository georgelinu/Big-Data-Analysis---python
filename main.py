import pandas as pd
import tweepy
import numpy as np
import seaborn as sns
import re, string, unicodedata
from string import punctuation
import nltk
nltk.download()
import contractions
import inflect
from nltk.stem import LancasterStemmer, WordNetLemmatizer
#import spacy
import tweepy

from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords  # import the stopwords collection from nltk.corpus
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud #, stopwords
import sklearn
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


consumer_key = "*************************************"
consumer_secret = "*************************************"
access_token = "*************************************"
access_token_secret = "*************************************"


#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

#auth.set_access_token(access_token, access_token_secret)

#api = tweepy.API(auth, wait_on_rate_limit=True)
#api = tweepy.API(auth)

#name = "WhiteHouse"

#tweetCount = 1500

#results = api.user_timeline(id=name, #count=tweetCount)

#for tweet in results:
 # print(tweet.text)


f=open("tweetFile.doc","r")
text1=f.read()
#print(text1)

#convert to lower case
input_str = text1.lower()
#print(input_str)

#remove white space
input_str = input_str.strip()
#input_str
result1 = re.sub(r"http\S+", "", input_str) #remove https from the text
#print(result1)
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(result1)
#text2 = ' '.join(str(x) for x in tokens )
result = [i for i in tokens if not i in stop_words]
#print (result)
result = [word for word in result if word.isalpha()]
print(result)

#word stemming
ps = PorterStemmer() 

for w in result: 
	print(w, " : ", ps.stem(w)) 


#Text visulization
#generate visulization for text1
wordcloud = WordCloud(stopwords=stop_words,background_color='white').generate(TreebankWordDetokenizer().detokenize(result))
plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis('off')
plt.savefig('wordc')
#plt.show()


#vectorize the text i.e. convert the strings to numeric features
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(result)

true_k = 10 #clustering steps
cls = MiniBatchKMeans(n_clusters = true_k, random_state = True).fit(features)

cls.labels_
cls.predict(features)

Kmeans_kwargs = {"init": "random","n_init":20, "max_iter": 300 , "random_state":42,}
#sum squared error
sse = []
for k in range(1,11):
    kmeans = KMeans(n_clusters = k, **Kmeans_kwargs)
    kmeans.fit(features)
    sse.append(kmeans.inertia_)   # The lowest SSE value
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.title('Elbow Method')
plt.xlabel("Number of Clusters")
plt.ylabel('SSE')
plt.savefig('elbow')
#plt.show()


print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
  print ("Cluster %d:" % i, end = '')
  for ind in order_centroids[i, :10]:
     print (' %s' % terms[ind], end = '')
  print()
  print()
print()

 