import pandas as pd
source_path = "../nyc-parking-tickets/Parking_Violations_Issued_-_Fiscal_Year_2015.csv"



for i,chunk in enumerate(pd.read_csv(source_path, chunksize=40000, dtype=dtypes)):
    chunk.to_csv('../tmp/split_csv_pandas/chunk{}.csv'.format(i), index=False)