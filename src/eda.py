import pandas as pd 

df = pd.read_csv("data/Resume.csv")

print("Shape:", df.shape)
print("\nKolom:", df.columns.tolist())
print("\nDistribusi kategori:")
print(df['Category'].value_counts())
print("\nContoh Resume_str:")
print(df['Resume_str'][0][:500])
