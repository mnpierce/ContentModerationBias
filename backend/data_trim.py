import pandas as pd

df = pd.read_csv('csv_data/full_train.csv')

# Remove rows where 'other_race_or_ethnicity' is NULL (NaN)
# Doesnt matter which label is used for this purpose, as they have all or none
df = df.dropna(subset=['other_race_or_ethnicity'])

# Remove rows where 'toxicity_annotator_count' is less than 10
df = df[df['toxicity_annotator_count'] >= 10]

df.to_csv('csv_data/trimmed_train.csv', index=False)