import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score,accuracy_score


data = pd.read_csv("C:\\Users\\Sumai\\OneDrive\\Desktop\\spam\\spam.csv",encoding='latin')

data.rename(columns={'v1':'Class','v2':'Text'},inplace=True)
data['numClass'] = data['Class'].map({'ham':0, 'spam':1})
data['Count']=0
for i in np.arange(0,len(data.Text)):
    data.loc[i,'Count'] = len(data.loc[i,'Text'])

print("Unique values in the Class set: ", data.Class.unique())

ham  = data[data.numClass == 0]
ham_count  = pd.DataFrame(pd.value_counts(ham['Count'],sort=True).sort_index())
print("Number of ham messages in data set:", ham['Class'].count())
spam = data[data.numClass == 1]
spam_count = pd.DataFrame(pd.value_counts(spam['Count'],sort=True).sort_index())
print("Number of spam messages in data set:", spam['Class'].count())


stopset = set(stopwords.words("english"))

vectorizer = CountVectorizer(stop_words=stopset,binary=True)
vectorizer = CountVectorizer()
    
X = vectorizer.fit_transform(data.Text)
y = data.numClass
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, train_size=0.70, random_state=None)
print("\n")
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
print("\n")



objects = ('Multi-NB','SVM','KNN', 'RF', 'AdaBoost')

def train_classifier(clf, X_train, y_train):    
    clf.fit(X_train, y_train)


def predict_labels(clf, features):
    return(clf.predict(features))
A = MultinomialNB()
B=  LinearSVC()
C = KNeighborsClassifier()
D = RandomForestClassifier()
E = AdaBoostClassifier()


clf = [A,B,C,D,E]
acc_score = [0,0,0,0,0]


for a in range(0,5):
    print(objects[a])
    train_classifier(clf[a], X_train, y_train)
    y_pred = predict_labels(clf[a],X_test)
    pred_val = f1_score(y_test, y_pred)
    acc_score[a]=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    print("F1 Score")
    print(pred_val)
    print("Accuracy in %:")
    print(acc_score[a] * 100)
    print("\n")

y_pos = np.arange(len(objects))
y_val = [ x for x in acc_score]
plt.bar(y_pos,y_val, align='center', alpha=0.7, color='black')
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy Score')
plt.title('Accuracy of Models')
plt.show()