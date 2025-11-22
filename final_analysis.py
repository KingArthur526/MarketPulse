
try:
    from sentiment_analyzer import SentimentAnalyzer
    print(" Import works")
except Exception as e:
    print(f" Import failed: {e}")
    exit()


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
