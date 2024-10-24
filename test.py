import pandas as pd

df = pd.read_csv('model_evaluation_results.csv')


# Define weights for each metric. You can adjust these based on what you feel is more important.
weights = {
    'mean_cosine_similarity': 0.25,  # Higher is better
    'silhouette_score': 0.25,        # Higher is better
    'dbcv_score': 0.2,               # Higher is better
    'noise_percentage': -0.1,        # Lower is better (negative weight -- minimize noise)
    'average_max_probability': 0.15, # Higher is better
    'mean_stability': 0.05           # Higher is better
}

# Normalize the data so all metrics are on the same scale (between 0 and 1)
for metric in weights.keys():
    if weights[metric] > 0:
        # For positive metrics (where higher is better), normalize between 0 and 1
        df[metric + '_normalized'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    else:
        # For negative metrics (where lower is better), normalize inversely
        df[metric + '_normalized'] = (df[metric].max() - df[metric]) / (df[metric].max() - df[metric].min())

# Calculate the composite score as a weighted sum of all normalized metrics
df['composite_score'] = 0
for metric, weight in weights.items():
    df['composite_score'] += weights[metric] * df[metric + '_normalized']

# Sort the models based on the composite score
df_sorted = df.sort_values(by='composite_score', ascending=False)

# Get the top model with the highest composite score
top_model = df_sorted.iloc[0]

# Print the model name
print("Top model:", top_model['model_name'])