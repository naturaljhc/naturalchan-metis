from flask import Flask
from flask import request, render_template
import numpy as np
import pandas as pd
import scipy
import spacy 

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')

#def spacy_tokenizer(sentence):
#    all_tokens = nlp(sentence)
#    tokens = [token.lemma_.lower().strip() for token in all_tokens if (token.pos_ in {"NOUN", "PROPN", "VERB"} and token.is_quote == False and len(token) > 1)]
#    lemmatized = [word.strip("\"'") for word in tokens if word not in stopwords]
#    return lemmatized

stopwords = nlp.Defaults.stop_words

import context_api
from context_api import cosine_sim, recommend_posts, corpus, cv, nmf, dtm

@app.route("/", methods = ["GET", "POST"])
def recommend():
    x_input, recommendations = recommend_posts(request.args)
    return render_template('index.html', 
            x_input = x_input,
            recommendations = recommendations)

if __name__ == '__main__':
    app.run(debug = True)
