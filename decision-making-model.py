from datetime import datetime, timedelta
import pandas as pd
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm

import nltk
from nltk.corpus import stopwords

# import word-2-vec model
from gensim.models import Word2Vec

#for visualizing data
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

#for clustering the data
from sklearn.decomposition import PCA

#using TSNE rather than PC -> preservies relationships between data pounts in a lower-dimensional space
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score, euclidean_distances
from sklearn.metrics import mutual_info_score

import math

import torch
from torch.nn.functional import cosine_similarity as torch_cos_sim
import umap

from sklearn.mixture import GaussianMixture

from gensim.models import FastText
import fasttext
from sentence_transformers import SentenceTransformer, util

#for accessing file paths
import os

#for hierarchical clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score

import random

from transformers import pipeline

import hdbscan
from hdbscan import validity_index

from collections import Counter

import io

import tempfile

from sklearn.preprocessing import normalize

from sklearn.metrics.pairwise import cosine_similarity

import openai

def build_collection(data):
    collection = [] #collection of words
    counter = 0 
    unique_genres = set()
    
    for index, sentence in data.items():
        if pd.notna(sentence):
            counter += 1
            word_list = sentence.split(",")  #split the list of words by separate names
            collection.append(word_list)
            for words in word_list:
                unique_genres.add(words)

    return unique_genres

def findOptimalCombination():

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

    return top_model['n_neighbors'], top_model['min_cluster_size']


def evaluateMetrics(n_neighbors, min_cluster_size, umap_results, clusterer, labels):

    #calculate cosine similarity ans silouette score-- for the data representation
    # Calculate silhouette score
    sil_score = silhouette_score(umap_results, labels)
    
    # Compute cosine similarity for all pairs 
    similarity_matrix = np.dot(umap_results, umap_results.T)  # This gives cosine similarities
    mean_cosine_similarity = np.mean(similarity_matrix)

    # Percentage of noise points
    noise_ratio = np.sum(labels == -1) / len(labels) * 100
    
    #calculate stability scores
    stability_scores = clusterer.cluster_persistence_
    mean_stability = np.mean(stability_scores)

    membership_probabilities = clusterer.probabilities_

    # Ensure membership_probabilities is 2D (if needed)
    if membership_probabilities.ndim == 1:
        # If 1D, reshape to (n_samples, 1)
        membership_probabilities = membership_probabilities.reshape(-1, 1)
    
    # Calculate metrics (this is an example, update based on your actual needs)
    average_max_probability = np.mean(np.max(membership_probabilities, axis=1))

    #calculate density-based clustering validation to evaluate the quality of density-based clusters
    dbcv_score = validity_index(umap_results, labels)

    metrics_df = pd.DataFrame({
        'n_neighbors': [n_neighbors],
        'min_cluster_size': [min_cluster_size],
        'silhouette_score': [sil_score],
        'mean_cosine_similarity': [mean_cosine_similarity],
        'noise_percentage': [f'{noise_ratio:.2f}'],
        'mean_stability': [mean_stability],
        'average_max_probability': [average_max_probability],
        'dbcv_score': [f'{dbcv_score:.4f}']
    })

    # Write DataFrame to CSV (appending to avoid overwriting)
    metrics_df.to_csv('model_evaluation_results.csv', mode='a', index=False, header=not pd.io.common.file_exists('model_evaluation_results.csv'))

#allows for cluster fine tuning for sparse genre categories in clusters
def subcluster_large_clusters(umap_results, labels, genre_df, genre_embeddings, genre_list, min_song_size=20, output_dir='playlists/'):

    # Ensure the output directory exists -- if not, creates it
    os.makedirs(output_dir, exist_ok=True)

    #make a genre_results.csv file with the genres in each generated playlist
    genre_results_file = os.path.join(output_dir, "genre_results.csv")
    genre_results = []

    # Dictionary that will hold the final subcluster results and stats list
    subcluster_results = {}
    stats_list = []

    # Replace spaces with underscores in genre_df['Genres']
    genre_df['Genres'] = genre_df['Genres'].str.replace(' ', '_')

    # Replace NaN or non-string values with an empty string
    genre_df['Genres'] = genre_df['Genres'].fillna('').astype(str)

    # Get rows where 'Genres' is not empty
    genre_df = genre_df[genre_df['Genres'] != ""]

    #performs a deep copy of dataframe for further alterations
    genre_df = genre_df.copy()

    # Create a mapping of each unique genre to its main cluster label
    genre_to_cluster = {genre: label for genre, label in zip(genre_list, labels)}

    for cluster_label in set(labels):

        # If the cluster is noise -> ignore
        if cluster_label == -1:
            continue
    
        # Filter points in the current main cluster and their genres
        cluster_points = umap_results[labels == cluster_label]
        cluster_genres = [genre for genre, label in genre_to_cluster.items() if label == cluster_label]

        #get optimal threshold whicu considers both cosine similarity and length of cluster points
        similarity_threshold = max(0.7, min(1.0, len(cluster_points) * 0.01))

        # Filter songs in genre_df that contain any of these genres
        cluster_genre_df = genre_df[
            genre_df['Genres'].apply(
                lambda x: isinstance(x, str) and any(g in x.split(',') for g in cluster_genres)
            )
        ]

        

        # Check if cluster_genre_df is empty, skip if so
        if cluster_genre_df.empty:
            continue

        
        # Calculate centroid
        centroid = np.mean(cluster_points, axis=0)
        
        # Calculate average distance to centroid
        distances = euclidean_distances(cluster_points, centroid.reshape(1, -1))
        avg_distance = np.mean(distances)

        #check if min similarity is less than threshold -> get subclusters -- min_song_size = 5 since that is 2 greater than n_components
        if avg_distance > 3 and len(cluster_genre_df) > min_song_size:
            # Set n_components to the minimum of 5 or the number of features (to avoid excessive reduction)
            n_components = min(5, len(cluster_points), cluster_points.shape[1])

            # PCA for further dimensionality reduction
            pca_sub = PCA(n_components=n_components, random_state=42)
            sub_pca_embeddings = pca_sub.fit_transform(cluster_points)

            # UMAP with adjusted parameters for subclustering
            umap_sub_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, spread=1.5, metric='cosine')
            
            if len(sub_pca_embeddings) >= umap_sub_reducer.n_neighbors:  # Ensure UMAP can process the data
                sub_umap_results = umap_sub_reducer.fit_transform(sub_pca_embeddings)
            else:
                print(f"Skipping UMAP for cluster {cluster_label} due to insufficient data points.")
                continue

            # HDBSCAN with adjusted min_cluster_size and min_samples for subclustering
            min_cluster_size_base = 5  # Use a higher base for subclustering
            min_samples_base = 3  # Raise the base value to reduce fragmentation

            # Adjust based on cluster size, but avoid making them too small
            min_cluster_size = max(min_cluster_size_base, int((labels == cluster_label).sum() * 0.1))  # 10% of cluster size
            min_samples = max(min_samples_base, int((labels == cluster_label).sum() * 0.05))  # 5% of cluster size

            # Cap the values to avoid excessive constraints
            min_cluster_size = min(min_cluster_size, 50)
            min_samples = min(min_samples, 20)

            # Use HDBSCAN with the refined parameters
            sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=0.01)
            sub_labels = sub_clusterer.fit_predict(sub_umap_results)

            # Store subcluster labels in the results dictionary
            subcluster_results[cluster_label] = sub_labels

            # Analyze each subcluster
            for sub_label in set(sub_labels):
                if sub_label == -1:  # Ignore noise in subclusters
                    continue
                
                # Identify points belonging to the current subcluster
                subcluster_points_indices = np.where(sub_labels == sub_label)[0]

                # Get genres associated with these points
                subcluster_genres = [
                    cluster_genres[i] for i in subcluster_points_indices if i < len(cluster_genres)
                ]

                
                # Filter songs associated with this subcluster
                subcluster_songs = cluster_genre_df[
                    cluster_genre_df['Genres'].apply(
                        lambda x: isinstance(x, str) and any(g in x.split(',') for g in subcluster_genres)
                    )
                ]

                #save playlist CSV for the subcluster
                playlist_filename = f"playlist_results_{cluster_label}_{sub_label}.csv"
                playlist_path = os.path.join(output_dir, playlist_filename)
                subcluster_songs.to_csv(playlist_path, index=False)

                # Append genres to genre_results for this playlist
                genre_results.append({
                    "main_cluster": cluster_label,
                    "sub_cluster": sub_label,
                    "genres": ', '.join(subcluster_genres)
                })

                # Calculate average upload date
                avg_upload_date = subcluster_songs['Added At'].apply(lambda x: datetime.fromisoformat(x))
                avg_upload_date = avg_upload_date.max().strftime('%Y-%m-%d')

                # Append statistics for each subcluster
                stats_list.append({
                    'main_cluster': cluster_label,
                    'sub_cluster': sub_label,
                    'song_count': len(subcluster_songs),
                    'max_upload_date': avg_upload_date,
                    "genres": ', '.join(subcluster_genres)
                })

        else:
            # For clusters with high similarity (no need to subclusters), save the entire cluster as one playlist
            playlist_filename = f"playlist_results_{cluster_label}_NA.csv"
            playlist_path = os.path.join(output_dir, playlist_filename)
            cluster_genre_df.to_csv(playlist_path, index=False)

            # Append genres to genre_results for this playlist
            genre_results.append({
                "main_cluster": cluster_label,
                "sub_cluster": "N/A",
                "genres": ', '.join(cluster_genres)
            })

            #compute average upload date
            avg_upload_date = cluster_genre_df['Added At'].apply(lambda x: datetime.fromisoformat(x)).max().strftime('%Y-%m-%d')

            stats_list.append({
                'main_cluster': cluster_label,
                'sub_cluster': 'N/A',  # No subclusters for smaller clusters
                'song_count': len(cluster_genre_df),
                'max_upload_date': avg_upload_date,
                "genres": ', '.join(cluster_genres)
            })

    # Save genre results to a separate CSV file
    pd.DataFrame(genre_results).to_csv(genre_results_file, index=False)

    # Convert stats list to DataFrame
    stats_df = pd.DataFrame(stats_list)

    return subcluster_results, stats_df


def computeCombinations(embeddings):
    #this is to investigate with different values
    params_grid = {
        'n_neighbors': [10, 20, 30],
        'min_cluster_size': [15, 20, 30]
    }

    pca = PCA(n_components=50, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)

    #iterate through all possible combinations -> this has a time complexity of O(n^3) which isnt very optimal but lets keep it for now
    for n_neighbors in params_grid['n_neighbors']:
        for min_cluster_size in params_grid['min_cluster_size']:
            
            umap_reducer = umap.UMAP(n_neighbors=n_neighbors, metric='cosine', min_dist=0, random_state=42, spread=20)
            umap_results = umap_reducer.fit_transform(pca_embeddings)

            umap_results = np.array(umap_results, dtype=np.float64)

            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5, cluster_selection_epsilon=0)
            labels = clusterer.fit_predict(umap_results)

            #call the evaluate metrics method to compute different metrics and save it as a csv file
            evaluateMetrics(n_neighbors, min_cluster_size, umap_results, clusterer, labels)


def generateTitle(genres):

    with open('key.txt', 'r') as file:
        openai.api_key = file.read()

    # Define the prompt as a system/user message
    messages = [
    {"role": "user", "content": (
        "Classify these genres into one general genre category, choosing from common music genres such as Pop, Rock, Hip-Hop, Jazz, Country, Classical, Electronic, R&B, Reggae, Dance, Folk, Blues, Latin, Reggaeton, Lo-fi, or similar. "
        f"{genres}. "
        "Answer with one specific genre name only."
    )}
]
    # Send the messages to ChatGPT API
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use "gpt-3.5-turbo" if desired
        messages=messages,
        max_tokens=5  # Limit tokens since the response is expected to be one word
    )
    # Extract and print the result
    return response['choices'][0]['message']['content'].strip()


def generate_playlists(primary_percentile=90, secondary_range=(40, 90), 
                       min_song_count=25):

    os.makedirs('generated_playlists/', exist_ok=True)

    #find date form 1 year ago
    date_threshold = datetime.now() - timedelta(days=365)

    # date_threshold = date_threshold.strftime("%Y-%m-%d")

    results_df = pd.read_csv('subcluster_statistics.csv')

    results_df['max_upload_date'] = pd.to_datetime(results_df['max_upload_date'])

    #cut off list of cluster playlists that are old
    results_df = results_df.loc[results_df['max_upload_date'] >= date_threshold]

    #further data preprocessing stage that will filter through songs and check for if characteristics are found
    category_features = {
        "Pop": {"liveness": 0.2, "acousticness": 0.6, "tempo": 120},
        "Hip-Hop": {"liveness": 0.4, "acousticness": 0.1, "tempo": 90},
        "Rock": {"liveness": 0.3, "acousticness": 0.2, "tempo": 130},
        "Jazz": {"liveness": 0.4, "acousticness": 0.7, "tempo": 80},
        "Classical": {"liveness": 0.1, "acousticness": 0.9, "tempo": 70},
        "Electronic": {"liveness": 0.2, "acousticness": 0.1, "tempo": 128},
        "R&B": {"liveness": 0.3, "acousticness": 0.5, "tempo": 85},
        "Reggae": {"liveness": 0.3, "acousticness": 0.6, "tempo": 75},
        "Country": {"liveness": 0.3, "acousticness": 0.8, "tempo": 100},
        "Dance": {"liveness": 0.2, "acousticness": 0.2, "tempo": 128},
        "Folk": {"liveness": 0.3, "acousticness": 0.9, "tempo": 100},
        "Blues": {"liveness": 0.4, "acousticness": 0.8, "tempo": 75},
        "Latin": {"liveness": 0.3, "acousticness": 0.4, "tempo": 100},
        "Reggaeton": {"liveness": 0.3, "acousticness": 0.3, "tempo": 90},
        "Lo-fi": {"liveness": 0.5, "acousticness": 0.8, "tempo": 70}
    }

    preprocessing = {}

    #tolerance levels for outliers in list of songs
    liveness_tolerance = 0.2
    acousticness_tolerance = 0.2
    tempo_tolerance = 20

    for index, row in results_df.iterrows():

        category = generateTitle(row['genres'])
        print(category)
        
        category_values = category_features.get(category)

        if not category_values:
            continue

        if pd.isna(row['sub_cluster']):
            playlist_file = f"playlists/playlist_results_{int(row['main_cluster'])}_NA.csv"
            temp_playlists = pd.read_csv(playlist_file)
        else:
            playlist_file = f"playlists/playlist_results_{int(row['main_cluster'])}_{int(row['sub_cluster'])}.csv"
            temp_playlists = pd.read_csv(playlist_file)

        filtered_songs = temp_playlists[
            (temp_playlists['Liveness'].between(category_features[category]['liveness'] - liveness_tolerance, category_features[category]['liveness'] + liveness_tolerance)) &
            (temp_playlists['Acousticness'].between(category_features[category]['acousticness'] - acousticness_tolerance, category_features[category]['acousticness'] + acousticness_tolerance)) &
            (temp_playlists['Tempo'].between(category_features[category]['tempo'] - tempo_tolerance, category_features[category]['tempo'] + tempo_tolerance))
        ]

        # Check if row['sub_cluster'] is NaN, and update only if not NaN
        sub_cluster_value = int(row['sub_cluster']) if not pd.isna(row['sub_cluster']) else row['sub_cluster']

        # Update song_count only if sub_cluster values match, and handle NaN values
        results_df.loc[
            (results_df['main_cluster'] == int(row['main_cluster'])) & 
            ((pd.isna(results_df['sub_cluster']) & pd.isna(row['sub_cluster'])) | 
            (results_df['sub_cluster'] == sub_cluster_value)), 
            'song_count'] = len(filtered_songs)
        
        if category in preprocessing:
            df1 = pd.read_csv(preprocessing[category])

            # Append temp_playlists to df1
            df1 = df1._append(filtered_songs, ignore_index=True)

            # Sort by 'Added At' in descending order
            df1 = df1.sort_values(by='Added At', ascending=False)

            # Reset index if needed (optional)
            df1.reset_index(drop=True, inplace=True)

            df1.to_csv(preprocessing[category], index=False)

        else:
            preprocessing[category] = playlist_file

        filtered_songs.to_csv(playlist_file, index=False)

    # Calculate percentiles for song counts
    song_count_percentiles = results_df['song_count'].quantile([primary_percentile / 100, 
                                                                secondary_range[0] / 100, 
                                                                secondary_range[1] / 100])

    primary_threshold = song_count_percentiles.iloc[0]
    secondary_low = song_count_percentiles.iloc[1]
    secondary_high = song_count_percentiles.iloc[2]

    count = 0

    playlists = {}

    #iterate through each entry in subcluster_statistics
    for genre, file_path in preprocessing.items():
        df = pd.read_csv(file_path)
        song_count = len(df)

        playlist_file = f'generated_playlists/playlist_{count}.csv'

        #playlists with majority of songs
        if song_count >= primary_threshold:
            primary_size = min(100, song_count)
            
            if primary_size == 100:
                #get first 70 playlists from most recently added
                temp_playlists1 = df.iloc[:70]

                #get a random 30 other songs
                temp_playlists2 = (df.iloc[70:]).sample(n=30, random_state=42) 

                # Reset the index of the DataFrame
                new_playlist = pd.concat([temp_playlists1, temp_playlists2], ignore_index=True)
                new_playlist.reset_index(drop=True, inplace=True)  # Ensure a clean index for the file

                df = new_playlist

            # Write to CSV
            df.to_csv(playlist_file, index=False)  # No ignore_index here

            playlist_title = f"Because you mostly listen to \"{genre}\" genre of songs"
            playlists[playlist_title] = playlist_file

            count += 1

        #playlists with sufficient # of songs but not primary
        elif secondary_low <= song_count <= secondary_high and song_count >= min_song_count:           
            df.to_csv(playlist_file, index=False) 
            playlist_title = f"You might also like \"{genre}\" style of music"

            count += 1
        
            playlists[playlist_title] = playlist_file

    orig_df = pd.read_csv('liked_songs.csv')

    orig_df['Added At'] = pd.to_datetime(orig_df['Added At'])

    # Ensure date_threshold is timezone-aware (e.g., UTC)
    date_threshold = pd.to_datetime(date_threshold).tz_localize('UTC')

    #get the subset dataframe that contains songs from the last year
    subset_orig_df = orig_df.loc[orig_df['Added At'] >= date_threshold]

    # splice each artist from artist names
    subset_orig_df['Individual Artists'] = subset_orig_df['Artist Name(s)'].str.split(',')
    exploded_df = subset_orig_df.explode('Individual Artists')

    # Identify the top 3 most listened-to artists
    top_artists = exploded_df['Individual Artists'].value_counts().head(3).index
    
    for artist in top_artists:
        playlist_title = f"Songs from {artist}"
        subset_artist = orig_df[orig_df['Artist Name(s)'].str.contains(artist)].head(100)
        #must contain at least 10 songs
        if len(subset_artist) >= 10:
            playlist_file = f'generated_playlists/playlist_{count}.csv'
            subset_artist.to_csv(playlist_file, index=False) 
            playlists[playlist_title] = playlist_file
            count += 1

    return playlists



#find data points for each subcluster o compute most popuar playlists
def dataPointsPerSubcluster(liked_songs):
    
    #group by Main Cluster nd Subcluster, and calculate song count and average date
    subcluster_stats = liked_songs.groupby(['main_cluster', 'subcluster']).agg(
        song_count = ('song', 'count'),
        avg_date = ('date_liked', 'mean')
    ).reset_index()

    #sort values for easier interpretation, with most recent dates and largest song counts
    subcluster_stats = subcluster_stats.sort_values(
    by=['main_cluster', 'song_count', 'avg_date'], 
    ascending=[True, False, False]
    )


    # Display results for each main cluster and its subclusters
    for main_cluster, group in subcluster_stats.groupby('main_cluster'):
        print(f"\nMain Cluster: {main_cluster}")
        for _, row in group.iterrows():
            print(f"  Subcluster: {row['subcluster']}")
            print(f"    Song Count: {row['song_count']}")
            print(f"    Average Date: {row['avg_date']}")


df = pd.read_csv('liked_songs.csv')

#build unique collection of words and preprocess
unique_collection_of_words = build_collection(df['Genres'])
unique_collection_of_words = [genre.replace(' ', '_') for genre in unique_collection_of_words]

print(unique_collection_of_words)

#convert genre data to vector embeddings 
model = SentenceTransformer('fine_tuned_genre_sbert')
#model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings1 = model.encode(unique_collection_of_words)

# use PCA to reduce dimensionality
pca = PCA(n_components=5, random_state=42)
pca_embeddings = pca.fit_transform(embeddings1)

# apply UMAP on the PCA-reduced data
umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, spread=5, metric='cosine', random_state=42)
umap_results = umap_reducer.fit_transform(pca_embeddings)

# cluster using HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=3, cluster_selection_epsilon=0)
labels = clusterer.fit_predict(umap_results)

sil_score = silhouette_score(umap_results, labels)
print(f"silhouette score: {sil_score}")

noise_ratio = np.sum(labels == -1) / len(labels) * 100
print(f"noise ratio: {noise_ratio}")


#for visualizing initial HBDSCAN clustering (before finetuning)
unique_labels = np.unique(labels)

# Define a list of markers (shapes) and colors
markers = ['o', 's', '^', 'P', '*', 'X', 'D']  # Customize this list with more shapes if needed
colors = plt.cm.get_cmap('tab10', len(unique_labels))  # Use a colormap with a fixed number of colors

#to show graph window on screen
plt.switch_backend('TkAgg')

# Plot each cluster with a unique color and shape
plt.figure(figsize=(10, 7))

for i, label in enumerate(unique_labels):
    # Get all points belonging to the current cluster
    cluster_mask = (labels == label)

    # Choose color and marker for the cluster
    color = colors(i)
    marker = markers[i % len(markers)]  # Cycle through the marker list if clusters exceed marker options
    
    # Handle noise (-1 label) separately (often gray and 'x' marker)
    if label == -1:
        plt.scatter(umap_results[cluster_mask, 0], umap_results[cluster_mask, 1],
                    c='gray', marker='x', s=50, label='Noise')
    else:
        plt.scatter(umap_results[cluster_mask, 0], umap_results[cluster_mask, 1],
                    c=[color], marker=marker, s=50, label=f'Cluster {label}')

# Add legend and titles
plt.legend()
plt.title('Clusters with Unique Colors and Shapes')
plt.show()

# Convert embeddings to a dictionary mapping genres to their embeddings
embeddings = {genre: embedding for genre, embedding in zip(unique_collection_of_words, embeddings1)}

# peform subclustering for clusters and map each song to that subcluster
subcluster_results, stats_df = subcluster_large_clusters(umap_results, labels, df, embeddings, unique_collection_of_words)

stats_df.to_csv("subcluster_statistics.csv", index=False)

#generate playlists based on statistics received
playlists = generate_playlists()

print(playlists)




