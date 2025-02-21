import json
import plotly
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from nltk.corpus import wordnet as wn

synsets = wn.synsets("dog")
print(synsets)


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:////Users/susiecrone/Documents/Project_2_Data_Science/data/data.db')
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load("/Users/susiecrone/Documents/Project_2_Data_Science/models/trained_model.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # First graph: Bar Chart
    bar_chart = {
        'data': [
            {
                'x': genre_names,
                'y': genre_counts.tolist(),
                'type': 'bar'
            }
        ],
        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {'title': "Count"},
            'xaxis': {'title': "Genre"}
        }
    }
    
    # Second graph: Pie Chart
    pie_chart = {
        'data': [
            {
                'values': genre_counts.tolist(),
                'labels': genre_names,
                'type': 'pie'
            }
        ],
        'layout': {
            'title': 'Percentage Distribution of Message Genres'
        }
    }
    
    # Combine both graphs into one list
    graphs = [bar_chart, pie_chart]
    
    # Create an ID for each graph for rendering in the template
    ids = ["graph-{}".format(i) for i in range(len(graphs))]
    
    # Convert the graphs list into JSON for Plotly
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render the master template with graphs and ids
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
   