import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Sentiment.csv')

dataset.candidate.value_counts().plot(kind = 'pie', autopct = '%1.0f%%')


twit_sent = dataset.groupby(['candidate', 'sentiment']).sentiment.count().unstack()
twit_sent.plot(kind = 'bar')

dataset.sentiment.value_counts().plot(kind = 'pie', autopct = '%1.0f%%', colors = ['red', 'blue', 'green'])

dataset = dataset.drop(dataset[dataset.sentiment == "Neutral"].index)
sent_map = {"Positive": 1, "Negative": 0}
dataset["sentiment"] = dataset["sentiment"].map(sent_map)
X = dataset["text"]
y = dataset["sentiment"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.feature_extraction.text import TfidfVectorizer
sent_tfidf = TfidfVectorizer(max_df = 0.7, min_df = 2, stop_words = 'english')

X_train = sent_tfidf.fit_transform(X_train).toarray()
X_test = sent_tfidf.transform(X_test).toarray()

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

pred_test_nb = nb_model.predict(X_test)
print(accuracy_score(y_test, pred_test_nb))


from sklearn.neighbors import KNeighborsClassifier
kneig_model = KNeighborsClassifier(n_neighbors = 5)
kneig_model.fit(X_train, y_train)

pred_test_kn = kneig_model.predict(X_test)
print(accuracy_score(y_test, pred_test_kn))


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_model.fit(X_train, y_train)

pred_test_rf = rf_model.predict(X_test)
print(accuracy_score(y_test, pred_test_rf))

twt = ['All #GOP candidates want to reduce taxes while #Huckabee wants to legalize prostitution and drugs so we can tax it. #GOPDebate']
X_twt = sent_tfidf.transform(twt).toarray() 
predict_twt = rf_model.predict(X_twt)
print(predict_twt)

























