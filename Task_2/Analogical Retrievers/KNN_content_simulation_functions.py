import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# import pickle
import json

train_df = pd.read_csv("task2/analogy/50klangbind_imgvid_embeddings.csv")

def find_closest_likes(df, input_company_name,input_likes,k=5):
    matching_rows = df[df['inferred company'] == input_company_name]

    if matching_rows.empty:
        print(f"No matching rows found for {input_company_name}")
        return None

    df_new = matching_rows.copy()

    df_new['likes_diff'] = abs(df['likes'] - input_likes)

    df_sorted = df_new.sort_values(by='likes_diff')

    if(len(df_sorted)>=k):
        return df_sorted[:k]
    else:
        return df_sorted

# with open('companies.pkl', 'rb') as file:
#     companies_arr = pickle.load(file)

with open('companies.json', 'r') as json_file:
    companies_arr = json.load(json_file)

def filter_date_top_k_likes(df, likes, input_date, inp_unseen_brand, k=5):
    input_date = datetime.strptime(input_date, '%Y-%m-%d %H:%M:%S')
    six_months = timedelta(days=30*6)
    start_date = input_date - six_months
    end_date = input_date
    df['date'] = pd.to_datetime(df['date'])
    
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if filtered_df.empty:
        print(f"No matching rows found for {input_date}")
        return None
    
    companies=[]
    for row in companies_arr:
        if inp_unseen_brand in row:
            companies=row
    
    filtered2_df = filtered_df[filtered_df['inferred company'].isin(companies)]
    
    filtered2_df['likes_distance'] = abs(filtered2_df['likes'] - likes)
    top_k_df = filtered2_df.nsmallest(k, 'likes_distance')
    
    return top_k_df.drop(columns=['likes_distance'])