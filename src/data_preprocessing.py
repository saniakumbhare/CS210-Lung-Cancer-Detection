import pandas as pd
import sqlite3

# Load the dataset
df = pd.read_csv('../data/raw/lung_cancer.csv')

# Data cleaning
df.fillna(df.mean(), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
df.drop_duplicates(inplace=True)
df['GENDER'] = df['GENDER'].str.upper()

# Data transformation
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 30, 60, 100], labels=['Young', 'Middle-aged', 'Old'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['AGE', 'SMOKING', 'ALCOHOL_CONSUMING']] = scaler.fit_transform(df[['AGE', 'SMOKING', 'ALCOHOL_CONSUMING']])

# Save the cleaned data
df.to_csv('../data/processed/lung_cancer_cleaned.csv', index=False)

# Store the data in SQLite database
conn = sqlite3.connect('../lung_cancer.db')
df.to_sql('lung_cancer_data', conn, if_exists='replace', index=False)
conn.close()
