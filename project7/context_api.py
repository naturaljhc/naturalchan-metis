import pickle5 as pkl5
import pickle as pkl
import numpy as np
import pandas as pd
import scipy
import spacy
import praw
from sqlalchemy import create_engine

nlp = spacy.load('en_core_web_sm')

#def spacy_tokenizer(sentence):
#    all_tokens = nlp(sentence)
#    tokens = [token.lemma_.lower().strip() for token in all_tokens if (token.pos_ in {"NOUN", "PROPN", "VERB"} and token.is_quote == False and len(token) > 1)]
#    lemmatized = [word.strip("\"'") for word in tokens if word not in stopwords]
#    return lemmatized

engine = create_engine('sqlite:///models/posts.db')
conn = engine.connect()
df = pd.read_sql('SELECT * FROM posts', con = engine)
corpus = pd.read_sql('SELECT * FROM corpus', con = engine)['0']
dtm = pd.read_sql('SELECT * FROM dtm', con = engine)
conn.close()
engine.dispose()

with open("models/cv.pkl", "rb") as file_name:
    cv = pkl5.load(file_name)
with open("models/nmf.pkl", "rb") as file_name:
    nmf = pkl5.load(file_name)

def cosine_sim(a,b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    else:
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def recommend_posts(post_url):
    reddit = praw.Reddit(
        client_id = "sJWEgXc5LkKbsymp1BKSyA",
        client_secret = "AvxFCAQL4sChsnCUK9HfPeYUMzw3zg",
        user_agent = "testbot"
    )
    x_input = post_url.get("Post Title or URL", 0)
    if x_input == 0:
        return (x_input, None)
    try:
        post_title = reddit.submission(url = x_input).title
    except:
        # post_title = x_input
        return (x_input, "Invalid")
    X_test = cv.transform([post_title]).toarray()
    dtm_test = pd.DataFrame(X_test, columns = cv.get_feature_names_out())
    sps_test = scipy.sparse.csr_matrix(dtm_test)
    post = np.array(pd.DataFrame(nmf.transform(sps_test))).flatten()
    distances = []
    for i in range(dtm.shape[0]):
        distances.append(cosine_sim(post, dtm.iloc[i,:]))
    distances = pd.Series(distances).sort_values(ascending = False)
    
    top_5_index = distances[:5].index
    top_5_posts = corpus[top_5_index]
    submission_list = reddit.info(fullnames = ["t3_" + df["id"][i] for i in top_5_index])
    top_5_urls = []
    for submission in submission_list:
        top_5_urls.append(submission.shortlink)
    recommendations = dict(zip(top_5_posts, top_5_urls))
    return (x_input, recommendations)
