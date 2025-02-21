import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

import pickle
from sqlalchemy import create_engine

# Define the URL regex outside, so it can be accessed by the tokenize function
url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """
    Load data from SQLite database.
    
    Args:
        database_filepath (str): Path to SQLite database file.
    
    Returns:
        tuple: Features (X) and target variables (y).
    """
    engine = create_engine('sqlite:////Users/susiecrone/Documents/Project_2_Data_Science/data/data.db')
    df = pd.read_sql("SELECT * FROM data", engine)
    X = df.message.values
    y = df.loc[:, 'related':'direct_report'].values
    return X, y

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Flag if the first word in a sentence is a verb.
    """
    def starting_verb(self, text):
        """
        Check if the first word in a sentence is a verb.
        
        Args:
            text (str): Input text.
        
        Returns:
            bool: True if the first word is a verb, else False.
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        """
        Fit method for scikit-learn model.
        """
        return self

    def transform(self, X):
        """
        Extract starting verbs.
        
        Args:
            X (pd.Series): Series of text data.
        
        Returns:
            pd.DataFrame: Dataframe containing boolean values flagging starting verbs.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    """
    Tokenize input text and replace URLs with placeholders.
    
    Args:
        text (str): Input text.
    
    Returns:
        list: List of cleaned tokens.
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Build machine learning pipeline using GridSearchCV.
    
    Returns:
        GridSearchCV: Grid search model pipeline.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__min_samples_split': [2, 3, 4]       
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, x_test, y_test, category_names):
    """
    Evaluate the model's performance, printing classification report.
    
    Args:
        model: Trained machine learning model.
        x_test (array-like): Test features.
        y_test (array-like): True labels.
        category_names (list): Category names.
    """
    y_pred = model.predict(x_test)  # Ensure y_pred is defined before using it
    y_test_bin = (y_test > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    # Generate the classification report
    report = classification_report(y_test_bin, y_pred_bin, target_names=category_names)
    print(report)

def save_model(model, model_filepath):
    """
    Save trained model to pickle file.
    
    Args:
        model: Trained machine learning model.
        model_filepath (str): Path to save pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Execute training pipeline with function.
    - Load data
    - Build model
    - Train model
    - Evaluate model
    - Save model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)  # Adjusted to match return values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, Y.columns)  # Assuming category names are from Y DataFrame

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
