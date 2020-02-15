# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Arguments:
        - string : filename for the database
        
    Output:
        no output, only a db database file will be created for the input filename and DataFrame
    """
    
    # load data from database
    engine = create_engine('sqlite:///etl-pipeline.db')
    df = pd.read_sql_table('etl-pipeline', engine)
    X = df['message'] 
    y = df.drop(columns=['message', 'genre'], axis=1) 
    target_names = [column for column in y.columns if column != 'message']
    
    return X, y, target_names


def tokenize(text):
    """
    Arguments:
        - string : text to be tokenized
        
    Output:
        - string : tokenized text
    """
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model():
    """
    Arguments:
        
    Output:
        - GridSearchCV model
    """
    
    # pipeline for the disaster response tokens
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1, n_estimators=50, oob_score = True)))
    ])
    
    # paramgrid
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 100, 500, 1000),
    'tfidf__use_idf': (True, False),
    "clf": [RandomForestClassifier()],
         "clf__n_estimators": [10, 100, 250],
         "clf__max_depth":[8],
         "clf__random_state":[42]
    }

    # gridsearch pipeline with paramgrid
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Arguments:
        - GridSearchCV : trained model
        - df : DataFrame of the X_test
        - df : DataFrame of the Y_test
        - string : string array of the 36 categories names
        
    Output:
        - no return, prints the classification report
    """
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Arguments:
        - GridSearchCV : trained model
        - string : filepath of the to saving model
   
    Output:
        - no return, saves the model as .pkl
    """
    pickle.dump(model, open(model_filepath + '/clf', 'wb'))


def main():
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
