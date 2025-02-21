import json
import plotly
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Determining whether the first word in a sentence is a verb.
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
            pos_tags = nltk.pos_tag(word_tokenize(sentence))
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
        Apply to extract starting verbs.
        
        Args:
            X (pd.Series): Series of str data.
        
        Returns:
            pd.DataFrame: Dataframe containing boolean values flagging starting verbs.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

app = Flask(__name__)

def tokenize(text):
    """
    Tokenize and lemmatize input text.
    
    Args:
        text (str): Input text.
    
    Returns:
        list: List of cleaned tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    return clean_tokens

# Load data
db_path = 'sqlite:////Users/susiecrone/Documents/Project_2_Data_Science/data/data.db'
engine = create_engine(db_path)
df = pd.read_sql_table('data', engine)

# Load model
model = joblib.load("/Users/susiecrone/Documents/Project_2_Data_Science/models/trained_model.pkl")

@app.route('/')
@app.route('/index')
def index():
    """
    Render the visualizations in index page.
    
    Returns:
        str: Rendered HTML template.
    """
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    bar_chart = {
        'data': [{'x': genre_names, 'y': genre_counts.tolist(), 'type': 'bar'}],
        'layout': {'title': 'Distribution of Message Genres', 'yaxis': {'title': "Count"}, 'xaxis': {'title': "Genre"}}
    }
    
    pie_chart = {
        'data': [{'values': genre_counts.tolist(), 'labels': genre_names, 'type': 'pie'}],
        'layout': {'title': 'Distribution of Message Genres as a Percentage'}
    }
    
    graphs = [bar_chart, pie_chart]
    ids = ["graph-{}".format(i) for i in range(len(graphs))]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    """
    Render the model predictions based on user input.
    
    Returns:
        str: Rendered HTML template.
    """
    query = request.args.get('query', '')
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    return render_template('go.html', query=query, classification_result=classification_results)

def main():
    """
    Run the Flask app.
    """
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
