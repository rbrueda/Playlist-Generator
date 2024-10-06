
#todo: goal - finish this part tonight so i can move onto different playlist cateogries --> popularity, release dates, artists name (should be simular to word to vec? -- but i think it isnt really -- we need to find the most popular)

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
            word_list = sentence.split(", ")  #split the list of words by separate names
            collection.append(word_list)
            for words in word_list:
                unique_genres.add(words)

    return collection

#todo: print all results in a text file
def evaluateMetrics(model_name, embedding_file, genre_vectors):

    #load bin file made
    model = fasttext.load_model(embedding_file)

    #apply UMAP to reduce dimensionality for model -- helps shape data better for clustering
    umap_reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=42)
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

df = pd.read_csv('sampleSongs.csv')

#here will list the columns to find relevant information from 
metricColumns = ['artistName', 'albumPopularity', 'albumReleaseDate', 'artistGenres']

#make a new dataframe with just these columns
sub_df = df[metricColumns]

#create a collection of words from album genres
collection_of_words = build_collection(sub_df['artistGenres'])

#flatten the list of lists into a single list
flat_genre_list = [genre for sublist in collection_of_words for genre in sublist]

print(f"length of genres flat: {len(flat_genre_list)}")

#since FastText model expects data to be either in a file or as an iteratable object -- doesnt accept list objects
# Replace spaces with underscores to preserve multi-word genres
flat_genre_list = [genre.replace(' ', '_') for genre in flat_genre_list]

#join all the values in the list as a space
genre_data_str = ' '.join(flat_genre_list)

# Use StringIO to create a file-like object in memory
genre_data_io = io.StringIO(genre_data_str)

#using custom model for this -- it is better since my data is highly genre specific and works and liked songs playlist can vary
# Train FastText on your dataset of genres -- custom model (less memory intensive might use this method later)

#parameters to try out for FastText model:
# vector_size -> use cosine similarity -- related gonres like "pop" and "dance pop" should have high cosine similarity since they sohlud be in close vector space
#window size -> for genres generally good to keep it low sine immediate subwords or closely related terms are often more important than distant ones
    # - check genre coherence 
#mincount -> filters out infrequent genres. If certain niche genres appear only once or twice in odel, might not contribute much to overall model
    # - check how removing rare genres affects the quality of your clustering
#Epochs -> controls the number pf training iterations 
    # more epochs -help model learm more about detailed relationships -> downfalls: can also lead to overfitting
    # - evaluate genre vector stability

#using grid search for parameter tuning
# i believe this means to explore different combinatins of parameter values

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



# model = FastText([list(genre) for genre in flat_genre_list], vector_size=100, min_count=1)

#pretrained fast text model -- will use this for now for simplicity (unless we need to fine tune later :) )
# model = fasttext.load_model('cc.en.300.bin')

# Get vectors for your genres
#genre_vectors = [fasttext_model.wv[genre] for genre in flat_genre_list]

genre_vectors = [model.get_word_vector(genre) for genre in flat_genre_list]


# #* increased perplexity to 30 -> to better reveal cluster structures
# tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)


# #todo: find most optimal perplexity vlues
# #perplexity: represents the # of nearest neighbors when constructing lower-dimensional embedding
# perplexities = [5, 10, 20, 30, 40, 50]
# tsne_results = []

# for perp in perplexities:
#     pca = PCA(n_components=50)  # Reduce to 50 dimensions first
#     pca_embeddings = pca.fit_transform(genre_embeddings)

#     # Apply t-SNE after PCA
#     tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perp, random_state=42)
#     tsne_results_pca = tsne.fit_transform(pca_embeddings)
#     tsne_results.append(tsne_results_pca)

plt.switch_backend('TkAgg')

# # Plot t-SNE results for different perplexity values
# plt.figure(figsize=(15, 10))
# for i, perp in enumerate(perplexities):
#     plt.subplot(2, 3, i+1)
#     plt.scatter(tsne_results[i][:, 0], tsne_results[i][:, 1], c='blue')
#     plt.title(f'Perplexity = {perp}')
# plt.tight_layout()
# plt.show()


#todo: experiment with different n_neightbor parameters


n_neighbors_values = [5, 10, 15, 30, 50, 75, 100]
#* note: keeping min_dist = 0.1 and random_state=42 by convention

# for neigh in n_neighbors_values:
#     umap_reducer = umap.UMAP(n_neighbors=neigh, min_dist=0.1, random_state=42)
#     umap_results = umap_reducer.fit_transform(genre_vectors)
    
#     # Plotting UMAP results
#     plt.scatter(umap_results[:, 0], umap_results[:, 1], c='blue', s=50)
#     plt.title(f"UMAP {neigh}")
#     plt.show()


#todo: use mathematical formula to find best formatting -- for now, find best value 

#what is noise?
#does noise show quality of clusters
#figure out best metrics to find most optimal combination

umap_reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=42)
umap_results = umap_reducer.fit_transform(genre_vectors)


#todo: figure out if changing min_cluster_size effects model
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

# #this is what we will start with for the number of clusters -- the number of broad categories
# genres = ["Hip Hop", "Pop", "Rock", "R&B", "Rap", "Country", "K-Pop", "Lo-fi", "Indie", "Metal", "Punk", "Jazz", "Classical"]

# #store the inertia value of each model and then plot it to visualize the result
# SSE = []

# #tsne scores
# silhouetteScores = []

# initial_cluster_range = range(2, int(math.sqrt(df.shape[0])), 2)

# for cluster in initial_cluster_range:
#     print(f"cluster#: {cluster}") 
#     kmeans = KMeans(n_clusters=cluster, init='k-means++')
#     labels = kmeans.fit_predict(umap_results)
#     SSE.append(kmeans.inertia_)
#     print(f"inertia #: {kmeans.inertia_}")
#     silhouetteScores.append(silhouette_score(umap_results, labels))
#     print(f"silhouette score: {silhouette_score(umap_results, labels)}")
#     print("-------------------------------------------")

# #to visual my data to check current progress
# plt.switch_backend('TkAgg')

# #see the elbow result visually using matplotlib
# frame = pd.DataFrame({'Cluster':initial_cluster_range, 'SSE':SSE, 'Silhouette':silhouetteScores})
# plt.figure(figsize=(12,6))


# # elbow plot
# plt.subplot(1, 2, 1)
# plt.plot(frame['Cluster'], frame['SSE'], marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')

# #silhouette score plot
# plt.subplot(1, 2, 2)
# plt.plot(frame['Cluster'], frame['Silhouette'], marker='o', color='orange')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Scores')

# plt.show()

# #find elbow and silhouette peak points
# elbow_point = np.argmin(np.diff(SSE))
# silhouette_peak = np.argmax(silhouetteScores)

# #find the zoomed range ariund better of the two (elbow or silhouette)

# #convert the indices of the elbow and silhouette peak points from initial broad range back into actual cluster numbers
# # n * 3 -> step of size of 3
# # range(2, 36, 3) = 2, 5, 8, 11, 14 .... 35
# #elbow_pint and silhouette back ae indices -> not actual cluster #s

# optimal_cluster_start = min(elbow_point, silhouette_peak) * 2 + 2
# optimal_cluster_end = max(elbow_point, silhouette_peak) * 2 + 5 # -> +5 allows for range to be slightly extended covering broader set of clusters

# print(f"Zooming in around cluster range {optimal_cluster_start} to {optimal_cluster_end}") 

# # Zoom in on the elbow region (based on observation of plots)
# zoomed_cluster_range = range(optimal_cluster_start, optimal_cluster_end, 1)  # Narrow down the range with step size of 1

# SSE_zoomed = []
# silhouetteScores_zoomed = []

# for cluster in zoomed_cluster_range:
#     kmeans = KMeans(n_clusters=cluster, init='k-means++')
#     labels = kmeans.fit_predict(umap_results)
#     SSE_zoomed.append(kmeans.inertia_)
#     silhouette_score_val = silhouette_score(umap_results, labels)
#     silhouetteScores_zoomed.append(silhouette_score_val)

#     print(f"cluster: {cluster}")
#     print(f"inertia #: {kmeans.inertia_}")
#     print(f"silhouette score: {silhouette_score_val}")
#     print("-------------------------------------------")


# #calculate second derivative for zoomed range
# first_derivative_zoomed = np.diff(SSE_zoomed)
# second_derivative_zoomed = np.diff(first_derivative_zoomed)
# threshold_zoomed = np.mean(np.abs(second_derivative_zoomed))

# #identify points with minimal change in second derivative
# #checks which second derivative values are less than mean, return the indexies of it and add it by the inital cluster number (to get actual cluster number)
# significant_points_zoomed = np.where(np.abs(second_derivative_zoomed) < threshold_zoomed)[0] + optimal_cluster_start


# #final visualization for zoomed range
# frame_zoomed = pd.DataFrame({'Cluster':range(optimal_cluster_start, optimal_cluster_end), 'SSE': SSE_zoomed, 'Silhouette': silhouetteScores_zoomed})
# plt.figure(figsize=(12,6))

# #elbow plot (zoomed)
# plt.subplot(1, 2, 1)
# plt.plot(frame_zoomed['Cluster'], frame_zoomed['SSE'], marker='o')
# plt.axvline(x=significant_points_zoomed[0], color='r', linestyle='--')  # Mark significant point
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.title('Zoomed Elbow Method')

# # Silhouette score plot (zoomed)
# plt.subplot(1, 2, 2)
# plt.plot(frame_zoomed['Cluster'], frame_zoomed['Silhouette'], marker='o', color='orange')
# plt.axvline(x=significant_points_zoomed[0], color='r', linestyle='--')  # Mark significant point
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Zoomed Silhouette Scores')

# plt.show()

# # Final suggestion for optimal cluster number
# print(f"Optimal cluster number based on second derivative: {significant_points_zoomed[0]}")

# #todo: tomorrow -> try to see what the value is when you just od 1 jump
# #todo: should we also consider the occurances of genre in liked songs

# #we can use fit_predict() to combine the fitting process with returning predictions in one step
# #random_state = 1 ensures that we get the same result if we run this model multiple times
# #use with initialization of kmeans++
# kmeans = KMeans(n_clusters=significant_points_zoomed[0], random_state=1, init='k-means++').fit_predict(umap_results)

# #to visual my data to check current progress
# plt.switch_backend('TkAgg')

# #list of colours
# colours = ["b", "g", "r", "c", "m", "y", "k", "w", "b", "g", "r", "c", "m", "y", "k", "w", "b", "g", "r", "c", "m", "y", "k", "w", "b", "g", "r", "c", "m", "y", "b", "g", "r", "c", "m", "b", "g", "r", "c", "m", "y", "k", "w", "b", "g", "r", "c", "m", "y", "k", "w", "b", "g", "r", "c", "m", "y", "k", "w", "b", "g", "r", "c", "m", "y", "b", "g", "r", "c", "m"]
# shapes = ["o", "*", "s", "p", "^", "d", "D", "x", "+", "P", "o", "s", "*", "p", "^", "d", "D", "x", "+", "P", "o", "*", "s", "p", "^", "d", "D", "x", "+", "P", "o", "*", "s", "p", "^", "d", "D", "o", "*", "s", "p", "^", "d", "D", "x", "+", "P", "o", "*", "s", "p", "^", "d", "D", "x", "+", "P", "o", "*", "s", "p", "^", "d", "D", "x", "+", "P", "o", "*", "s", "p", "^", "d", "D"]

# plt.figure(figsize = (30, 10))
# for i in range (0, significant_points_zoomed[0]):
#     print(i)
#     plt.plot(umap_results[kmeans==i, 0], umap_results[kmeans==i, 1], str(colours[i]+shapes[i]))

# plt.show()

# # #fit K-means model
# # # kmeans.fit(PCA_result)

# # #algorithm used for parititioning a dataset into a pre-defined number of clusters
# # # clusters = kmeans.predict(PCA_result)


# # # dendrogram = sch.dendrogram(sch.linkage(PCA_result, method='ward'))
# # # plt.show()

# # # hc = AgglomerativeClustering(n_clusters=len(genres),linkage='ward')

# # # y_hc = hc.fit_predict(PCA_result)

# # # plt.figure(figsize = (8, 5))
# # # plt.plot(PCA_result[y_hc==0, 0], PCA_result[y_hc==0, 1], "bs")
# # # plt.plot(PCA_result[y_hc==1, 0], PCA_result[y_hc==1, 1], "gs")
# # # plt.plot(PCA_result[y_hc==2, 0], PCA_result[y_hc==2, 1], "rs")
# # # plt.plot(PCA_result[y_hc==3, 0], PCA_result[y_hc==3, 1], "cs")
# # # plt.plot(PCA_result[y_hc==4, 0], PCA_result[y_hc==4, 1], "ms")
# # # plt.plot(PCA_result[y_hc==5, 0], PCA_result[y_hc==5, 1], "ys")
# # # plt.plot(PCA_result[y_hc==6, 0], PCA_result[y_hc==6, 1], "rs")
# # # plt.plot(PCA_result[y_hc==7, 0], PCA_result[y_hc==7, 1], "ks")
# # # plt.plot(PCA_result[y_hc==8, 0], PCA_result[y_hc==8, 1], "ws")
# # # plt.plot(PCA_result[y_hc==9, 0], PCA_result[y_hc==9, 1], "r^")
# # # plt.plot(PCA_result[y_hc==10, 0], PCA_result[y_hc==10, 1], "g^")
# # # plt.plot(PCA_result[y_hc==11, 0], PCA_result[y_hc==11, 1], "c^")
# # # plt.plot(PCA_result[y_hc==12, 0], PCA_result[y_hc==12, 1], "b^")
# # # plt.show()

# # #dataframe for PCA results and clusters
# TSNE_df = pd.DataFrame(TSNE_result, columns=['x_values', 'y_values'])
# TSNE_df['word'] = model.wv.index_to_key
# TSNE_df['cluster'] = kmeans

# # #look at the result of PCA dataframe
# # st.write(PCA_df)

# # #not sure how time consuming this is but ill do it like this: search for word in dataframe and add it to the cluster value

# #split the list of artistGenres from sub_df into lists of individual genres (in order to perform merge with PCA_df)
# sub_df['artistGenres'] = sub_df['artistGenres'].str.split(', ')

# #explode the list from sub_df into separate rows
# sub_df_exploded = sub_df.explode('artistGenres')

# # st.write(sub_df_exploded)

# #merge the exploded dataframe with PCA_df for the artist genre
# df_merged_genre = pd.merge(sub_df_exploded, TSNE_df, left_on='artistGenres', right_on='word', how='left')

# #iterate through each line of the cluster and print to a specific csv file
# for index, entry in df_merged_genre.iterrows():
#     # Check if the 'cluster' value is not NaN
#     if pd.notna(entry['cluster']):
#         # Open the file for writing
#         with open('samples/clusterRes' + str(entry['cluster']) + '.csv', 'a') as f:
#             # Convert the row (entry) to a DataFrame
#             entry_df = entry.to_frame().T

#             # Check if the file is empty to write the header
#             write_header = f.tell() == 0

#             # Write the DataFrame to the CSV file
#             entry_df.to_csv(f, index=False, header=write_header)

# generator = pipeline('text-generation', model='gpt2')

# #iterate through all the csv files and append the values to a set and write the results to the console
# for filename in os.listdir('samples'):
#     #get the full directory name
#     file_path = os.path.join('samples', filename)

#     #check if its a file and not directory
#     if os.path.isfile(file_path):
#         unique_genres = set()
#         print("\n------------------------")
#         print(filename)

#         df = pd.read_csv(file_path)
        
#         #iterate through all the entries in the dataframe and add the genre to a set
#         for index, entry in df.iterrows():
#             unique_genres.add(entry['artistGenres'])

#         sentence = "Create a playlist title for genres: "

#         #append all the unique genres to a sentence for a model
#         for genre in unique_genres:
#             print(genre)
#             sentence += genre + ", "

#         generated_text = generator(sentence, max_new_tokens = 10)
#         playlist_title = generated_text[0]['generated_text'].replace(sentence, '').strip()
#         print(f"playlist title: {playlist_title}")
#         print()

# #todo: use a transformer model to create concise, creative playlist titles for playlists