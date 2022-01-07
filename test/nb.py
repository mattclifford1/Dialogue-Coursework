import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
y = np.array([[1, 0], [1, 1], [1, 0], [0,1], [1, 1], [0,1]])

clf = MultinomialNB()
clf = BernoulliNB()
clf.fit(X, y)

print(MultinomialNB())
print(X)
print(clf.predict(X[2:3]))
print(clf.predict_log_proba(X[2:3]))
print(clf.predict_proba(X[2:3]))
