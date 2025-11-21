import pandas as pd
df = pd.read_csv("analyse.csv")
positive_df = df[df["sentiment"].str.lower() == "positive"]
negative_df = df[df["sentiment"].str.lower() == "negative"]
most_positive = positive_df["product name"].value_counts().head(10)
most_negative = negative_df["product name"].value_counts().head(10)
