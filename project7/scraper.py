#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import pandas as pd
import praw
import datetime as dt
from sqlalchemy import create_engine
from pmaw import PushshiftAPI
# from IPython.display import clear_output


# In[3]:

os.chdir('/home/chanjunho/reddit_context_bot')


# Connect to praw
reddit = praw.Reddit(
    client_id = "sJWEgXc5LkKbsymp1BKSyA",
    client_secret = "AvxFCAQL4sChsnCUK9HfPeYUMzw3zg",
    user_agent = "testbot"
)


# In[4]:


# Call PushShift API
api = PushshiftAPI()

# Scrape function
def psscrape_all(reddit, subreddit_name, after_time, before_time, corpus_df):
    """
    reddit - PRAW Reddit object
    subreddit_name - The name of the subreddit to scrape
    after_time - Beginning time to scrape (epoch time)
    before_time - Ending time to scrape (epoch time)
    corpus_df - DataFrame to save data in
    """
    
    # Gather all posts in the subreddit within the time frame using PushShift
    posts = api.search_submissions(subreddit = subreddit_name, 
                                   limit = None,
                                   after = after_time,
                                   before = before_time)
    
    # Create an id string for each post to gather upvote data
    # using PRAW (Reddit API)
    ids = ["t3_" + post["id"] for post in posts]
    
    # Go though each post and check the number of upvotes
    # If it has at least 1000 upvotes, add it to the corpus
    for submission in reddit.info(fullnames = ids):
        if submission.score >= 1000:
            corpus_df = corpus_df.append({"subreddit": subreddit_name,
                                          "title": submission.title, 
                                          "score": submission.score,
                                          "id": submission.id,
                                          "date": dt.datetime.fromtimestamp(submission.created).strftime('%d %B, %Y')},
                                         ignore_index = True)
            # Print each post that was added for debugging
            # print(submission.title)
    return corpus_df


# In[33]:


# Read old corpus so we can update it
# corpus_df = pd.read_csv('corpus.csv')
# We can also initialize an empty dataframe and start from scratch
# corpus_df = pd.DataFrame(columns = ["subreddit", "title", "score", "id", "date"])
engine = create_engine('sqlite:///models/posts.db')
conn = engine.connect()
corpus_df = pd.read_sql('SELECT * FROM posts', 
                        con = engine)

# List of subreddits to scrape from
subreddits = ["news", "worldnews"]

# We will scrape through each day
one_day = 86400

# Starting on the day determined by after_time and before_time
# before_time = int(dt.datetime(2010,1,2,0,0).timestamp())
# after_time = int(dt.datetime(2010,1,1,0,0).timestamp())

# end_time = int(dt.datetime.today().timestamp())
before_time = int(dt.datetime.strptime(corpus_df["date"].iloc[-1], '%d %B, %Y').timestamp()) - 2 * one_day
after_time = int(dt.datetime.strptime(corpus_df["date"].iloc[-1], '%d %B, %Y').timestamp()) - 3 * one_day

# We will loop through each day until the last desired date is reached
# which will be determined by end_time
end_time = int(dt.datetime.today().timestamp())


# In[34]:


# While we haven't reached the final date, we will loop through every
# single date and decrease before_time and after_time by 1 day after
# each loop
while after_time < end_time:
    try:
        # Print out what date is being scraped
        print(f"Currently scraping posts from {dt.datetime.fromtimestamp(after_time).strftime('%d %B, %Y')}.")
        # Go through every single subreddit in the list of subreddits
        for subreddit_name in subreddits:
            # Scrape the desired date using our scrape function
            corpus_df = psscrape_all(reddit, subreddit_name, after_time, before_time, corpus_df)
        # In case of duplicates from errors, we will drop the duplicates
        corpus_df = corpus_df.iloc[corpus_df["id"].drop_duplicates().index].reset_index(drop = True)
        # Print confirmation of completion and the updated number of rows in the corpus
        print(f"{dt.datetime.fromtimestamp(after_time).strftime('%d %B, %Y')} complete. Corpus contains {corpus_df.shape[0]} rows.")
        # Go back another day
        before_time += one_day
        after_time += one_day
    except:
        # If we have an error, we will redo the last date and try again
        # This is primarily in case there is a connection problem with
        # the PushShift API
        
        # Code currently works, but in case of a different error, an 
        # infinite loop may occur so this would allow us to identify
        # if debugging is required
        print("Error found, retrying last iteration")
        pass


# In[35]:


# engine = create_engine('sqlite:///posts.db', echo = False)
# conn = engine.connect()

corpus_df.to_sql('posts', con = engine, if_exists = 'replace', index = False)

conn.close()
engine.dispose()


# In[ ]:




