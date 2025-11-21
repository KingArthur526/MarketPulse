import pandas as pd

csv_source = " Path of the kaggle dataset"

csv_destination = "Path of the processed csv file"

df = pd.read_csv(csv_source)

sorted_df = df.sort_values(by = "Date", ascending= False)

chunked_df = sorted_df.sample(n= 300)
