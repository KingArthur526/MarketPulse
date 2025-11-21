import pandas as pd
df = pd.read_csv("analyse.csv")
final_df = df.read_csv("final_result.csv")
for row in df.iterrows:
  ratio= sentiment.count("positive")/sentiment.count("negative")
  if(ratio>1):
    final_df["action"]="buy"
  else if(ratio==1):
    continue
  else:
    final_df["action"]="sell"
  final_df["stock"]=row["Stock Name"]
  final_df["priority"]=ratio
most_positive=final_df["priority"].max()
most_negative=final_df["priority"].min()

sorted_df = final_df.sort(by = "ratio", ascending = False)
print(sorted_df)
