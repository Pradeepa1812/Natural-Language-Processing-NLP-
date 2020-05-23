import pandas as pd
messages=pd.read_csv('spams',sep='\t',names=['labels','message'])


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lem=WordNetLemmatizer()
corpus=[]

for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[lem.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
    
#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['labels'])
y=y.iloc[:,1].values

#train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)
    
#Training model using Naive BAyes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect=MultinomialNB().fit(X_train,y_train)
y_pred=spam_detect.predict(X_test)

#confusion_matrix
from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

#accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)