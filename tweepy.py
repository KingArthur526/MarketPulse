import tweepy
import pandas as pd

#consumer_key = "************" #Your API/Consumer key 
#consumer_secret = "*********" #Your API/Consumer Secret Key
#access_token = "***********"    #Your Access token key
#access_token_secret = "*************" #Your Access token Secret key

auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret,
    access_token, access_token_secret
)
Instantiate the tweepy API
api = tweepy.API(auth, wait_on_rate_limit=True)


search_query = "'Enter data here'-filter:retweets AND -filter:replies AND -filter:links"
no_of_tweets = 100

try:
    
    tweets = api.search_tweets(q=search_query, lang="en", count=no_of_tweets, tweet_mode ='extended')
    
    
    attributes_container = [[tweet.user.name, tweet.created_at, tweet.favorite_count, tweet.source, tweet.full_text] for tweet in tweets]

    
    columns = ["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"]
    
    
    tweets_df = pd.DataFrame(attributes_container, columns=columns)
except BaseException as e:
    print('Status Failed On,',str(e))
     
