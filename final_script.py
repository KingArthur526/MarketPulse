import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from twilio.rest import Client
import os



"""Loading and chunking the tweets"""

csv_source = "C:/Users/shrad/OneDrive/Desktop/MarketPulse/stock_tweets.csv"

df = pd.read_csv(csv_source)

sorted_df = df.sort_values(by = "Date", ascending= False)

chunked_df = sorted_df.sample(n= 500)

chunked_df.to_csv("C:/Users/shrad/OneDrive/Desktop/MarketPulse/clean_text2.csv", index= False)




"""FinBERT classes, initialization and test tweet"""



class SentimentAnalyzer:
    def __init__(self):
        print("Loading FinBERT model... This takes 2-3 minutes first time.")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        print("Model loaded successfully!")
    
    def analyze_single(self, text):
        """Analyze one tweet"""
        # Check if text is empty
        if not text or len(text.strip()) == 0:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            }
        
        # Convert text to model input
        inputs = self.tokenizer(text, return_tensors="pt", 
                                padding=True, truncation=True, max_length=512)
        
        # Get prediction
        outputs = self.model(**inputs)
        
        # Convert to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get the prediction
        prediction = torch.argmax(probs, dim=-1).item()
        labels = ['positive', 'negative', 'neutral']
        
        # Return result
        return {
            'sentiment': labels[prediction],
            'confidence': round(probs[0][prediction].item(), 3),
            'scores': {
                'positive': round(probs[0][0].item(), 3),
                'negative': round(probs[0][1].item(), 3),
                'neutral': round(probs[0][2].item(), 3)
            }
        }
    
    def analyze_batch(self, tweet_list):
        
        results = []
        total = len(tweet_list)
        for i, tweet in enumerate(tweet_list, 1):
            print(f"Processing tweet {i}/{total}...", end='\r')
            result = self.analyze_single(tweet)
            result['text'] = tweet
            results.append(result)
        print(f"\nCompleted analyzing {total} tweets!")
        return results


if __name__ == "__main__":
    print("Testing sentiment analyzer...")
    analyzer = SentimentAnalyzer()
    
    test_tweet = "Apple stock is soaring after great earnings! 📈"
    result = analyzer.analyze_single(test_tweet)
    
    print(f"\nTest Tweet: {test_tweet}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print("\nTest passed! ✓")

"""NLP Processing and Tweet Analysis"""

"""try:
    from sentiment_analyzer import SentimentAnalyzer
    print(" Import works")
except Exception as e:
    print(f" Import failed: {e}")
    exit()"""


try:
    analyzer = SentimentAnalyzer()
    print(" Analyzer created")
except Exception as e:
    print(f" Creation failed: {e}")
    exit()


try:
    result = analyzer.analyze_single("Test tweet")
    print(" Analysis works")
    print(f"Result: {result}")
except Exception as e:
    print(f" Analysis failed: {e}")
    exit()


try:
    import pandas as pd
    df = pd.DataFrame({'text': ['tweet1', 'tweet2']})
    print(" Pandas works")
except Exception as e:
    print(f" Pandas failed: {e}")
    exit()

print("\n" + "="*70)
print("basic tests passed")
print("="*70)




def analyze_tweet(text):
    
    return analyzer.analyze_single(text)

def analyze_dataframe(df, text_column='Tweet'):
    
    results = []
    for i, text in enumerate(df[text_column], 1):
        print(f"Processing {i}/{len(df)}...")
        result = analyzer.analyze_single(text)
        results.append(result)
    
    df = df.copy()
    df['sentiment'] = [r['sentiment'] for r in results]
    df['confidence'] = [r['confidence'] for r in results]
    return df

df = pd.read_csv("C:/Users/shrad/OneDrive/Desktop/MarketPulse/clean_text2.csv")

# DEMO

final_df = pd.DataFrame(df)

result_df = analyze_dataframe(final_df)

print("\nRESULTS:")
print("="*70)

drop_neutral = result_df[result_df['sentiment'] == 'neutral'].index

result_df.drop(drop_neutral, inplace=True)

grouped_df = result_df.groupby("Stock Name").agg({
    'sentiment': list,
    'confidence': list
})


print(grouped_df)

grouped_df.to_csv("C:/Users/shrad/OneDrive/Desktop/MarketPulse/grouped_result.csv")
print("="*70)




""" Sorting the sentimental data"""




df = pd.read_csv("C:/Users/shrad/OneDrive/Desktop/MarketPulse/grouped_result.csv")

import ast
df["sentiment"] = df["sentiment"].apply(ast.literal_eval)

rows = []

for stock, group in df.groupby("Stock Name"):

    all_labels = [label for lst in group["sentiment"] for label in lst]

    pos = sum(1 for x in all_labels if x == "positive")
    neg = sum(1 for x in all_labels if x == "negative")

    # avoid div by zero
    if neg == 0:
        ratio = float('inf')
    else:
        ratio = pos / neg

    if ratio > 1:
        action = "buy"
    elif ratio == 1:
        action = "hold"
    else:
        action = "sell"

    rows.append({
        "stock": stock,
        "pos": pos,
        "neg": neg,
        "ratio": ratio,
        "action": action
    })

final_df = pd.DataFrame(rows)

final_df = final_df.sort_values(by="ratio", ascending=False)

final_df.to_csv("C:/Users/shrad/OneDrive/Desktop/MarketPulse/final_result.csv", index=False)

print(final_df)




"""Sending results through Twilio sms service post verification of the phone number"""





df = pd.read_csv("C:/Users/shrad/OneDrive/Desktop/MarketPulse/final_result.csv")

first_row = df.iloc[0]
most_buyable = f"Most positive: {first_row['stock']} ({first_row['action']})"

last_row = df.iloc[-1]
most_sellable = f"Most negetive: {last_row['stock']} ({last_row['action']})"

sms_body = most_buyable + "\n\n"
sms_body += most_sellable
print(sms_body)

#twilio part

account_sid = "AC7fabd9629b8e50411c5482e65c621317"
auth_token = "1dbc639fe63408f3ca35aa72b01bc776"
client = Client(account_sid, auth_token)
account = client.api.accounts(account_sid).fetch()
print("Account name:", account.friendly_name)
message = client.messages.create(
    body=sms_body,
    from_="+12076107128",   #twilio
    to="+917892394912"      #number
)

print("SMS sent SID:", message.sid)
