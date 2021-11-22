import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from wordcloud import WordCloud
from collections import Counter

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

api = KaggleApi()
api.authenticate()
api.dataset_download_file('uciml/sms-spam-collection-dataset', file_name='spam.csv')

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
# print(df.shape)
# print(df.head())

# print(df.info())
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
# print(df.head())

df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
# print(df.head())

encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
# print(df.head())

# print(df.isnull().sum())

# print(df.duplicated().sum())
df.drop_duplicates(keep="first", inplace=True)
# print(df.duplicated().sum())

# print(df['target'].value_counts())
# plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
# plt.show()

nltk.download('punkt')

df['num_characters'] = df['text'].apply(len)
# print(df.head())

df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
# print(df.head())

df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
# print(df.head())

# print(df[['num_characters', 'num_words', 'num_sentences']].describe())
# print(df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe())
# print(df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe())

# sns.histplot(df[df['target'] == 0]['num_words'])
# sns.histplot(df[df['target'] == 1]['num_words'], color='red')
# plt.show()

# sns.pairplot(df, hue='target')
# plt.show()

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    list = []
    for i in text:
        if i.isalnum():
            list.append(i)

    text = list[:]
    list.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            list.append(i)

    text = list[:]
    list.clear()

    for i in text:
        list.append(ps.stem(i))

    return " ".join(list)


# print(transform_text(df['text'][100]))
df['transformed_text'] = df['text'].apply(transform_text)
# print(df.head())

# wc = WordCloud(width=500, height=500, min_font_size=10, background_color='black')
# spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
# plt.imshow(spam_wc)
# plt.show()
#
# wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
# spam_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
# plt.imshow(spam_wc)
# plt.show()

# ham_corpus = []
# for words in df[df['target'] == 0]['transformed_text'].tolist():
#     for word in words.split():
#         ham_corpus.append(word)
#
# # print(len(spam_corpus))
# sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0], pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()
#
# spam_corpus = []
# for words in df[df['target'] == 1]['transformed_text'].tolist():
#     for word in words.split():
#         spam_corpus.append(word)
#
# # print(len(spam_corpus))
# sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0], pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()

cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000) # MultinomialNB accuracy increases when max_feature = 3000 along with some minute changes in other algo

# scalar = MinMaxScaler()

# X = cv.fit_transform(df['transformed_text']).toarray()
X = tfidf.fit_transform(df['transformed_text']).toarray()
# X = scalar.fit_transform(X)
# print(X.shape)
y = df['target'].values
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model

# Naive bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_predGNB = gnb.predict(X_test)
print("GaussianNB:")
print(accuracy_score(y_test, y_predGNB))
print(confusion_matrix(y_test, y_predGNB))
print(precision_score(y_test, y_predGNB))

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_predMNB = mnb.predict(X_test)
print()
print("MultinomialNB:")
print(accuracy_score(y_test, y_predMNB))
print(confusion_matrix(y_test, y_predMNB))
print(precision_score(y_test, y_predMNB)) # on tfidf precision score is 1, in our case precision matters more than accuracy score

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_predBNB = bnb.predict(X_test)
print()
print("BernoulliNB:")
print(accuracy_score(y_test, y_predBNB))
print(confusion_matrix(y_test, y_predBNB))
print(precision_score(y_test, y_predBNB))

# Logistic regression
from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression(solver='liblinear', penalty='l1')
lrc.fit(X_train, y_train)
y_predLRC = lrc.predict(X_test)
print()
print("LogisticRegression:")
print(accuracy_score(y_test, y_predLRC))
print(confusion_matrix(y_test, y_predLRC))
print(precision_score(y_test, y_predLRC))

# SVC
from sklearn.svm import SVC

svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(X_train, y_train)
y_predSVC = svc.predict(X_test)
print()
print("SVC:")
print(accuracy_score(y_test, y_predSVC))
print(confusion_matrix(y_test, y_predSVC))
print(precision_score(y_test, y_predSVC))

# Decision tree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X_train, y_train)
y_predDTC = dtc.predict(X_test)
print()
print("DecisionTreeClassifier:")
print(accuracy_score(y_test, y_predDTC))
print(confusion_matrix(y_test, y_predDTC))
print(precision_score(y_test, y_predDTC))

# KNeighbor
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predKNC = knc.predict(X_test)
print()
print("KNeighborsClassifier:")
print(accuracy_score(y_test, y_predKNC))
print(confusion_matrix(y_test, y_predKNC))
print(precision_score(y_test, y_predKNC))

# Random forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50, random_state=2)
rfc.fit(X_train, y_train)
y_predRFC = rfc.predict(X_test)
print()
print("RandomForestClassifier:")
print(accuracy_score(y_test, y_predRFC))
print(confusion_matrix(y_test, y_predRFC))
print(precision_score(y_test, y_predRFC))

# Voting classifier:
from sklearn.ensemble import VotingClassifier

svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
voting = VotingClassifier(estimators=[('knc', knc), ('mnb', mnb), ('rfc', rfc), ('svc', svc), ('bnb', bnb)], voting='soft')
voting.fit(X_train, y_train)
y_predVOTE = voting.predict(X_test)
print()
print("VotingClassifier:")
print(accuracy_score(y_test, y_predVOTE))
print(precision_score(y_test, y_predVOTE))

# Stacking
# from sklearn.ensemble import StackingClassifier
#
# clf = StackingClassifier(estimators=[('knc', knc), ('mnb', mnb), ('rfc', rfc), ('svc', svc), ('bnb', bnb)], final_estimator=RandomForestClassifier())
# clf.fit(X_train, y_train)
# y_predCLF = voting.predict(X_test)
# print()
# print("Stacking:")
# print(accuracy_score(y_test, y_predCLF))
# print(precision_score(y_test, y_predCLF))

# import pickle
# pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
# pickle.dump(voting, open('model.pkl', 'wb'))

import joblib

joblib.dump(voting, 'model.pkl', compress=4)
joblib.dump(tfidf, 'vectorizer.pkl')
