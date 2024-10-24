
#todo: goal - finish this part tonight so i can move onto different playlist cateogries --> popularity, release dates, artists name (should be simular to word to vec? -- but i think it isnt really -- we need to find the most popular)
#todo: see if there is a way to look for user profile credentials

#* look at energetic count -- if they listen to upbeat songs recommend upbead songs playlist, else more relaxing
#* look at ho they have been recently listening to mostly and recommend playlist for them on that music artist

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

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import mutual_info_score

import math

import umap

from sklearn.mixture import GaussianMixture


#todo: instead of using word2Vec for data representation, using a larger, related dataset is more effective since word2Vec is better for larger datasets
from gensim.models import FastText
import fasttext
from sentence_transformers import SentenceTransformer

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

#function to create a collection of data
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


    print(f"number of unique genres is: {unique_genres}")
    return collection

def findOptimalCombination():
    #* goal: ensuring high similarity within clusters while maintaining distinct separation between clusters 
    #recommended metric ranking (for now): 
    # 1. mean cosine similarity -> directly measures how vectors of genres are within the same cluster
    # 2. silhouette score -> balances both intra- and inter-cluster metrics (make clear genre groupings)
    # 3. dbcv score -> since we are using HDBSCAM, this score is effective for assessing quality of clusters based on density and separation
    # 4. noise percentage -> important if you want to accoun for outliers or "noisy" genres
    # 5. average max probability -> not very relevant only for ensuring that clusters are assigend confidently to clusters
    # 6. mean stability -> not too important unless you expect your genre data to change significantly -- since we are re-evaluating metrics for new data anyways -- shouldnt be necessary only for narrow down combinations
    
    df = pd.read_csv('model_evaluation_results.csv')

    #filter from 27 combinations to 15 
    top_15_df = df.nlargest(15, 'mean_cosine_similarity')

    #filter from 15 combinations to 10
    top_10_df = top_15_df.nlargest(10, 'silhouette_score')

    #filter from 10 combinations to 6
    top_6_df = top_10_df.nlargest(6, 'dbcv_score')

    #filter from 6 combinations to 3
    top_3_df = top_6_df.nsmallest(3, 'noise_percentage')

    #filter from 3 combinations to 2
    top_2_df = top_3_df.nlargest(2, 'average_max_probability')

    #filter from 2 combinations to 1
    top_1_df = top_2_df.nlargest(1, 'mean_stability')



    return str(top_1_df['model_name'].iloc[0])





#todo: print all results in a text file
def evaluateMetrics(model_name, embedding_file, genre_vectors):

    #load bin file made
    model = fasttext.load_model(embedding_file)

    #apply UMAP to reduce dimensionality for model -- helps shape data better for clustering
    umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
    umap_results = umap_reducer.fit_transform(genre_vectors)

    #convert data with 64-bit precision
    umap_results = np.asarray(umap_results, dtype=np.float64)

    #todo: figure out if changing min_cluster_size effects model
    clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=30)
    labels = clusterer.fit_predict(umap_results)

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
        'model_name': [model_name],
        'silhouette_score': [sil_score],
        'mean_cosine_similarity': [mean_cosine_similarity],
        'noise_percentage': [f'{noise_ratio:.2f}'],
        'mean_stability': [mean_stability],
        'average_max_probability': [average_max_probability],
        'dbcv_score': [f'{dbcv_score:.4f}']
    })

    # Write DataFrame to CSV (appending to avoid overwriting)
    metrics_df.to_csv('model_evaluation_results.csv', mode='a', index=False, header=not pd.io.common.file_exists('model_evaluation_results.csv'))

df = pd.read_csv('liked_songs.csv')

#create a collection of words from album genres
collection_of_words = build_collection(df['Genres'])

#flatten the list of lists into a single list
flat_genre_list = [genre for sublist in collection_of_words for genre in sublist]

print(flat_genre_list)

#since FastText model expects data to be either in a file or as an iteratable object -- doesnt accept list objects
# Replace spaces with underscores to preserve multi-word genres
flat_genre_list = [genre.replace(' ', '_') for genre in flat_genre_list]

#join all the values in the list as a space
genre_data_str = ' '.join(flat_genre_list)

# Use StringIO to create a file-like object in memory
genre_data_io = io.StringIO(genre_data_str)

params_grid = {
    'vector_size': [50, 100, 200],
    'window_size': [3, 5, 7],
    'min_count': [1, 2, 5]
}

with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp_file:
    temp_file.write(genre_data_str)
    temp_file_path = temp_file.name  # Store the path to the temporary file

#iterate through all possible combinations -> this has a time complexity of O(n^3) which isnt very optimal but lets keep it for now
for vec_size in params_grid['vector_size']:
    for win_size in params_grid['window_size']:
        for min_count in params_grid['min_count']:
            model = fasttext.train_unsupervised(
                input=temp_file_path,
                model= "skipgram", #better for rare genres
                dim=vec_size,
                ws=win_size,
                minCount=min_count
            )

            model_name = f"fasttext_model_dim{vec_size}_ws{win_size}_min{min_count}"
            
            file_name = model_name + ".bin"

            #this helps with not having to continually trying to iterate through the function
            model.save_model(file_name)

            # Get the vector representation for each genre
            genre_vectors = [model.get_word_vector(genre) for genre in flat_genre_list]

            #call the evaluate metrics method to compute different metrics and save it as a csv file
            evaluateMetrics(model_name, file_name, genre_vectors)



model_name = findOptimalCombination()

model = fasttext.load_model(model_name+".bin")

genre_vectors = [model.get_word_vector(genre) for genre in flat_genre_list]

#* focusing on using 15 for a more local structure, meaning umap will prioritize grouping genres more closely related
#todo: seems like rap is repeated twice in my dataset -- todo: figure out the issue with this
#todo: experiement n_neighbours with values 10 and 5


umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
umap_results = umap_reducer.fit_transform(genre_vectors)

#todo: if the results arent very accurate, experiment with this value
clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=30)
labels = clusterer.fit_predict(umap_results)


maximum_cluster = 0

# Count the number of points in each cluster
cluster_counts = Counter(labels)

# Print the number of points per cluster
for cluster_label, count in cluster_counts.items():
    if cluster_label == -1:
        print(f'Noise points: {count}')
    else:
        print(f'Cluster {cluster_label}: {count} points')

        val = int(cluster_label)

        if val > maximum_cluster:
            maximum_cluster = val

print(f"total # of clusters: {maximum_cluster}")

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


umap_df = pd.DataFrame(umap_results, columns=['x_values', 'y_values'])

#todo: only select specific clusters based off datapoint count -- more points means more dense
#* maybe later i will even consider how new these songs were added(more newer means that user is more interested in this area of songs)


umap_df['word'] = flat_genre_list
umap_df['cluster'] = labels

# Group by 'genre' and take the first occurrence of each genre -- avoid duplicates 
umap_df_condensed = umap_df.groupby('word').first().reset_index()

#split the list of artistGenres from sub_df into lists of individual genres (in order to perform merge with PCA_df)
df['Genres'] = df['Genres'].str.split(',')

#explode the list from sub_df into separate rows
sub_df_exploded = df.explode('Genres')

#merge the exploded dataframe with PCA_df for the artist genre
df_merged_genre = pd.merge(sub_df_exploded, umap_df_condensed[['word', 'cluster']], left_on='Genres', right_on='word', how='left')

#* going ot print out the df_merged_genre to a csv file to trace the issue i am currently having with this :/
df_merged_genre.to_csv('merged.csv')

#iterate through each cluster and save the results in a csv
written_songs_per_cluster = {}

#todo: check what are the unique genres in each playlist

for index, entry in df_merged_genre.iterrows():
    # Check if the 'cluster' value is not NaN and not -1 for noise
    if pd.notna(entry['cluster']) and entry['cluster']:
        # Open the file for writing
        with open('samples/clusterRes' + str(entry['cluster']) + '.csv', 'a') as f:
            # Convert the row (entry) to a DataFrame
            entry_df = entry.to_frame().T

            # Check if the file is empty to write the header
            write_header = f.tell() == 0

            # Write the DataFrame to the CSV file
            entry_df.to_csv(f, index=False, header=write_header)

#iterate through all the csv files and append the values to a set and write the results to the console
for filename in os.listdir('samples'):
    #get the full directory name
    file_path = os.path.join('samples', filename)

    #check if its a file and not directory
    if os.path.isfile(file_path):
        unique_genres = set()
        print("\n------------------------")
        print(filename)

        df = pd.read_csv(file_path)
        
        #iterate through all the entries in the dataframe and add the genre to a set
        for index, entry in df.iterrows():
            unique_genres.add(entry['Genres'])



        #append all the unique genres to a sentence for a model
        for genre in unique_genres:
            print(genre)

        