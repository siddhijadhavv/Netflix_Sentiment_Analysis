# Importing the necessary libraries
import os
import argparse
import numpy as np
import pandas as pd
import re
import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report
import joblib
import model_dispatcher
import config

sw = stopwords.words('english')

def cleaning_data(df):
    # converting every word to lower case
    df.loc[: , 'review'] = df.loc[: , 'review'].apply(lambda x : x.lower()) 
    # Handling "@" characters if any
    df.loc[: , 'review'] = df.loc[: , 'review'].apply(lambda x : re.sub(r"@\S+" , "" , x))
    # removing punctuations
    df.loc[: , 'review'] = df.loc[: , 'review'].apply\
    (lambda x : x.translate(str.maketrans(dict.fromkeys(string.punctuation))))
    # joining back all the words into a sentence
    df.loc[: , 'review'] = df.loc[: , 'review'].apply\
    (lambda x : " ".join([word for word in x.split() if word not in (sw)]))
    return df

def lemmatization(df):
    lemmatized_review = []
    for i in df['review']:
        text = TextBlob(i)
        lemmatize=  []
        for j in text.words:
            word = Word(j)
            lemmatize.append(word.lemmatize())
        lemmatized_review.append(" ".join(lemmatize))
    df['lemmatized_review'] = lemmatized_review  
    return df

def run(model):
        # Reading the positvie text data
    pos_rev = pd.read_csv(config.POS_TRAINING_FILE , sep='\n' , header = None , encoding = 'latin-1')
        # adding a target column for positive text dataframe
    pos_rev['mood'] = 1.0
        # Renaming the column
    pos_rev = pos_rev.rename(columns = {0:'review'})

        # Reading the negative text data
    neg_rev = pd.read_csv(config.NEG_TRAINING_FILE  , sep='\n' , header = None , encoding = 'latin-1')
        # adding a target column from negative text dataframe
    neg_rev['mood'] = 0.0
        # Renaming the column
    neg_rev = neg_rev.rename(columns = {0:'review'})

        # Cleaning the data
    cleaning_data(pos_rev)
    cleaning_data(neg_rev)

        # Lemmatizing the data
    lemmatization(pos_rev)
    lemmatization(neg_rev)

        # Concatenating the pos and neg data
    com_rev = pd.concat([pos_rev, neg_rev], axis = 0).reset_index()

        # train test split
    X_train, X_test, y_train, y_test = train_test_split \
    (com_rev['lemmatized_review'].values, com_rev['mood'].values, test_size = 0.3, random_state = 101)

        # Creating train and test dataframes
    train_data = pd.DataFrame({'lemmatized_review': X_train, 'mood': y_train})
    test_data = pd.DataFrame({'lemmatized_review': X_test, 'mood': y_test})

        # Initialize the Vectorizer
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_data['lemmatized_review'])
    test_vectors = vectorizer.transform(test_data['lemmatized_review'])

        # Initializing the classifier, fitting and predicting on the test data
    if model == 'naive_bayes_gaussian' or model == 'naive_bayes_multinominal':
        classifier = model_dispatcher.model[model]
        classifier.fit(train_vectors.toarray(), train_data['mood'])
        pred = classifier.predict(test_vectors.toarray())
    else:
        classifier = model_dispatcher.model[model]
        classifier.fit(train_vectors, train_data['mood'])
        pred = classifier.predict(test_vectors)

        # Classification report
    report = classification_report(test_data['mood'], pred, output_dict = True)
    print(f"positive {report['1.0']['recall']}")
    print(f"negative {report['0.0']['recall']}")

        # Save the model
    joblib.dump(classifier, os.path.join(config.MODEL_OUTPUT, f"model_{model}.pkl"))
    joblib.dump(vectorizer, os.path.join(config.VECTORIZER_OUTPUT, f"vectorizer_{model}.pkl"))

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the argument for model selection
    parser.add_argument("--model", type = str)

    # read the arguments from the command line
    args = parser.parse_args()
    
    # run the fold specified by command line arguments
    run(model = args.model)