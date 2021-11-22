# SMS spam-or-ham
## Demo
link: https://spam-ham-sms.herokuapp.com/docs <br>
example: https://spam-ham-sms.herokuapp.com/spamorham/YOUR_MESSAGE
## Overview
This is a SMS spam classifier API built using fastAPI deployed on Heroku free dyno. The trained model uses [UCIML SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset). The model is a Voting classifier with K Neighbors, Multinomial Naive Bayes, Random Forest, SVC, and Bernoulli Naive Bayes as estimators with an **accuracy of 0.9783118405627198** and **precision of 1.0.**
## Approach
Naive Bayes is always favored for text-related tasks [(see)](https://www.quora.com/What-are-the-popular-ML-algorithms-for-email-spam-detection) however, I wanted to see if we combine various algorithms will the voted/stacked algorithm will be able to produce a better result than Naive Bayes.
<br>
### Steps:
1. Dataset is downloaded using Kaggle API
2. Unnamed columns are dropped and columns are renamed
3. The target column is encoded
4. Duplicate messages are removed
5. New columns containing tokenized word and sentence is created
6. Tokenization, Stopword removal, Stemming (using PorterStemmer), Vectorization  (using TfidfVectorizer) is done
7. Train and test split
8. Algorithms used:<br>
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
  - Logistic Regression
  - SVC (Support Vector Classifier)
  - Decision Tree
  - K Neighbors
  - Random Forest
9. Voting classifier with K Neighbors, Multinomial Naive Bayes, Random Forest, SVC, and Bernoulli Naive Bayes as estimators
10. Stacking Classifier with KNeighbors, Multinomial Naive Bayes, Random Forest, SVC and Bernoulli Naive Bayes as estimators and Random Forest as a final estimator
11. The voting classifier is pickled using joblib applying level 4 compression

### Improve existing model:
1. During vectorization, Multinomial Naive Bayes accuracy increases when max_feature = 3000 along with some minute changes in other algorithms
2. Scaling our features doesn't make any major improvements so I skip it
3. Voting classifier with K Neighbors, Multinomial Naive Bayes, Random Forest, SVC, and Bernoulli Naive Bayes as estimators and Multinomial Naive Bayes have a precision of 1.0
4. Voting classifier has an accuracy of 0.9783118405627198 where as Multinomial Naive Bayes has an accuracy of 0.9513481828839391 
<br> [See this for score and different config used](https://github.com/Maanaav/sms-spam-or-ham/blob/main/score.pdf)
