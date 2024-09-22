
#todo: find the dimension length from the data -- find the most optimal scale number using a mathematical equation
#todo: figure out the most optimal min_count number -- this value probably works best with 1, as we want all values to produce some unique value -- but im not sure how it will behave with "dance pop" versus "pop" -- similar words

#todo: figure out issue and propose solution 
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

import math

#for accessing file paths
import os

#for hierarchical clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score

import random

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

    print(f"number of genres: {len(unique_genres)}")
    print(unique_genres)


    return collection

def cosine_similarity(v1, v2):
    return dot(v1, v2) / (norm(v1) * norm(v2))


df = pd.read_csv('sampleSongs.csv')

#here will list the columns to find relevant information from 
metricColumns = ['artistName', 'albumPopularity', 'albumReleaseDate', 'artistGenres']

#make a new dataframe with just these columns
sub_df = df[metricColumns]

nltk.download('stopwords')

#gets all the stopwords in english
stop = set(stopwords.words("english"))

#create a collection of words from album genres
collection_of_words = build_collection(sub_df['artistGenres'])


#create a hashmap with storing the different sizes
models = {}

#list of genre pairs that are correlated -- used to calculate cosine distance
genre_pairs = [
    ('Pop', 'Dance Pop'),
    ('Pop', 'New Wave Pop'),
    ('Rock', 'Alternative Rock'),
    ('Rock', 'Classic Rock'),
    ('Hip Hop', 'Rap'),
    ('Hip Hop', 'Underground Hip Hop'),
    ('Hip Hop', 'Lgbtq+ Hip Hop'),
    ('Hip Hop', 'Political Hip Hop'),
    ('Jazz', 'Vocal Jazz'),
    ('Dream Pop', 'Modern Dream Pop'),
    ('Indie Pop', 'Indie Rock'),
    ('Indie Pop', 'Indie Poptimism'),
    ('Indie Pop', 'Indie Anthem-folk'),
    ('R&b', 'Chill R&b'),
    ('R&b', 'Dark R&b'),
    ('R&b', 'Neo Soul'),
    ('Pop', 'Pop House'),
    ('Pop', 'Pop R&b'),
    ('Dance Pop', 'Pop House'),
    ('Electronic', 'Progressive House'),
    ('Electronic', 'Electro House'),
    ('Electronic', 'Dubstep'),
    ('Electronic', 'Deep Tropical House'),
    ('Electronic', 'Electropop'),
    ('Electronic', 'Swedish Electropop'),
    ('Alternative Dance', 'Electropop'),
    ('Synthpop', 'Neo-synthpop'),
    ('Post-punk', 'Post-grunge'),
    ('Pop Rock', 'Alternative Rock'),
    ('Country', 'Classic Oklahoma Country'),
    ('Country', 'Modern Country Pop'),
    ('Country', 'Contemporary Country'),
    ('Country', 'Country Dawn'),
    ('Alternative Rock', 'Modern Alternative Rock'),
    ('Alternative Rock', 'Christian Alternative Rock'),
    ('Alternative Metal', 'Nu Metal'),
    ('Post-punk', 'New Wave'),
    ('New Wave', 'New Romantic'),
    ('Dream Pop', 'Chillwave'),
    ('Indie Game Soundtrack', 'Video Game Music'),
    ('Chillwave', 'Ambient Folk'),
    ('Lo-fi Rap', 'Sad Rap'),
    ('Sad Lo-fi', 'Sad Rap'),
    ('Synthpop', 'Electropop'),
    ('Synthpop', 'Swedish Electropop'),
    ('Synthpop', 'Electra'),
    ('Vocaloid', 'Japanese Teen Pop'),
    ('J-pop', 'Japanese Chillhop'),
    ('J-pop', 'J-poprock'),
    ('K-pop', 'K-pop Girl Group'),
    ('K-pop', 'K-pop Boy Group'),
    ('K-rap', 'Korean Pop'),
    ('J-rock', 'Visual Kei'),
    ('J-rock', 'Anime Rock'),
    ('Scandipop', 'Swedish Pop'),
    ('Scandipop', 'Swedish Electropop'),
    ('Vapor Pop', 'Vapor Soul'),
    ('Vapor Pop', 'Vaporwave'),
    ('Dark Clubbing', 'Hypertechno'),
    ('Talentschau', 'Talent Show'),
    ('Dance Rock', 'Pop Rock'),
    ('Modern Indie Pop', 'Modern Indie Folk'),
    ('Indie Rock Italiano', 'Italian Pop'),
    ('Indie Anthem-folk', 'Americana'),
    ('Big Room', 'Progressive Electro House'),
    ('Canadian Trap', 'Vancouver Indie'),
    ('Canadian Pop', 'Canadian Indie'),
    ('Canadian Electronic', 'Canadian Electropop'),
    ('Canadian Contemporary R&b', 'Canadian R&b'),
    ('French Hip Hop', 'Belgian Pop'),
    ('Belgian Pop', 'Dutch Pop'),
    ('Australian Dance', 'Australian Pop'),
    ('Australian Alternative Pop', 'Australian Hip Hop'),
    ('Pop Quebecois', 'French Pop'),
    ('Viral Pop', 'Social Media Pop'),
    ('Viral Trap', 'Trap Queen'),
    ('Dark R&b', 'Vapor Soul'),
]

#todo: find the most optimal vector_size number
vector_sizes = [20, 50, 100, 200, 500, 1000]

#try different vector sizes and compute silhouette score
for size in vector_sizes:
    model = Word2Vec(collection_of_words, vector_size=size, min_count=1)
    models[size] = model

#compute the cosine similarity
similarities = {}
for size, model in models.items():
    similarities[size] = []
    for genre1, genre2 in genre_pairs:
        if genre1 in model.wv:
            vec1 = model.wv[genre1]
        else:
            vec1 = None
        if genre2 in model.wv:
            vec2 = model.wv[genre2]
        else:
            vec2 = None
        
        #check if both vector values exists (there is an existent genre pair)
        if vec1 is not None and vec2 is not None:
            similarity = cosine_similarity(vec1, vec2)
            similarities[size].append(similarity)

#calculate the average cosine similarity across genre pairs
average_similarities = {size: sum(sim) / len(sim) for size, sim in similarities.items()}

#find the highest number to find the optimal vector size
optimal_size = max(average_similarities, key=average_similarities.get)
print(f"Optimal vector size: {optimal_size}")

#train a word2vec model
model = Word2Vec(collection_of_words, vector_size=optimal_size, min_count=1)

#create a keyed vector instance
keyed_vectors = model.wv

#vectorize the list of words
vectors = keyed_vectors[model.wv.index_to_key]
words = list(vectors)

#creates an instance of a 2d PCA
pca = PCA(n_components=2)

#todo: use TSNE instead of PCA
tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)

#fit the vectors in the PCA object created -- used for dimensionality reduction
PCA_result = pca.fit_transform(vectors)

TSNE_result = tsne.fit_transform(vectors)

#this is what we will start with for the number of clusters -- the number of broad categories
genres = ["Hip Hop", "Pop", "Rock", "R&B", "Rap", "Country", "K-Pop", "Lo-fi", "Indie", "Metal", "Punk", "Jazz", "Classical"]

#store the inertia value of each model and then plot it to visualize the result
SSE = []

#tsne scores
sihouetteScores = []

cluster_range = range(2, int(math.sqrt(df.shape[0])))

for cluster in cluster_range:
    print(f"cluster#: {cluster}") 
    kmeans = KMeans(n_clusters=cluster, init='k-means++')
    labels = kmeans.fit_predict(TSNE_result)
    SSE.append(kmeans.inertia_)
    print(f"inertia #: {kmeans.inertia_}")
    sihouetteScores.append(silhouette_score(TSNE_result, labels))
    print(f"silhouette score: {silhouette_score(TSNE_result, labels)}")
    print("-------------------------------------------")
    #todo: look at results-> if good start working on next clustering problem

#to visual my data to check current progress
plt.switch_backend('TkAgg')

#see the elbow result visually using matplotlib
frame = pd.DataFrame({'Cluster':range(2, int(math.sqrt(df.shape[0]))), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#elbow method -> measures the sum of squared distances between data points and their assigned cluster centroids
#narrow down the range of values where ebow method comes into play to get the most optimal result
#use the second derivative to see how much the rate of change changes over time -- find out when the change is minimal

#calculate first derivative
first_derivative = np.diff(SSE)

#calculate second derivative
second_derivative = np.diff(first_derivative)

#use the mean of the second derivative values to find the "normal" level of fluctuation of data
threshold = np.mean(np.abs(second_derivative))
significant_points = []


#iterate through the second derivative values
for i in range(1, len(second_derivative)):
    #if the second derivative value is less than the normal level, then data points are closer to their respective centroids
    if abs(second_derivative[i]) < threshold:
        significant_points.append(cluster_range[i+1]) #we use i+1 because when you calculate the first derivative, it reduces the length od the original array by 1

#silhouette score -> measures how similar data points are within their assigned cluster compared to other clusters
#afterwards -> find the most optimal silhouette score depending on results from elbow score

#iterate through the elements from elbow method in the most optimal silhouette score from there
significant_scores = []
for cluster in significant_points:
    significant_scores.append((cluster, sihouetteScores[cluster-2]))

print(significant_scores)

#convert the significant scores to a DataFrame for finding best cluster optimally
#2d dataframe
significant_df = pd.DataFrame(significant_scores, columns = ['Cluster', 'Silhouette Score'])

#find best cluster based on silhouette cluster - find the max silhouette score in df
#higher silhouette score -> measures how similar an object is to its own cluster compared to other clusters
best_cluster = significant_df.loc[significant_df['Silhouette Score'].idxmax()]

print(f"Best cluster# based on silhouette score and elbow method: {best_cluster['Cluster']}")

#we can use fit_predict() to combine the fitting process with returning predictions in one step
#random_state = 1 ensures that we get the same result if we run this model multiple times
#use with initialization of kmeans++
kmeans = KMeans(n_clusters=int(best_cluster['Cluster']), random_state=1, init='k-means++').fit_predict(TSNE_result)

#to visual my data to check current progress
plt.switch_backend('TkAgg')

#list of colours
colours = ["b", "g", "r", "c", "m", "y", "k", "w"]
shapes = ["o", "*", "s", "p", "^", "v", "<", ">", "d", "D", "x", "X", "+", "P"]

plt.figure(figsize = (8, 5))
for i in range (0, int(best_cluster['Cluster'])):
    plt.plot(TSNE_result[kmeans==i, 0], TSNE_result[kmeans==i, 1], str(colours[random.randint(0, len(colours)-1)]+shapes[random.randint(0, len(shapes)-1)]))

plt.show()

# #fit K-means model
# # kmeans.fit(PCA_result)

# #algorithm used for parititioning a dataset into a pre-defined number of clusters
# # clusters = kmeans.predict(PCA_result)


# # dendrogram = sch.dendrogram(sch.linkage(PCA_result, method='ward'))
# # plt.show()

# # hc = AgglomerativeClustering(n_clusters=len(genres),linkage='ward')

# # y_hc = hc.fit_predict(PCA_result)

# # plt.figure(figsize = (8, 5))
# # plt.plot(PCA_result[y_hc==0, 0], PCA_result[y_hc==0, 1], "bs")
# # plt.plot(PCA_result[y_hc==1, 0], PCA_result[y_hc==1, 1], "gs")
# # plt.plot(PCA_result[y_hc==2, 0], PCA_result[y_hc==2, 1], "rs")
# # plt.plot(PCA_result[y_hc==3, 0], PCA_result[y_hc==3, 1], "cs")
# # plt.plot(PCA_result[y_hc==4, 0], PCA_result[y_hc==4, 1], "ms")
# # plt.plot(PCA_result[y_hc==5, 0], PCA_result[y_hc==5, 1], "ys")
# # plt.plot(PCA_result[y_hc==6, 0], PCA_result[y_hc==6, 1], "rs")
# # plt.plot(PCA_result[y_hc==7, 0], PCA_result[y_hc==7, 1], "ks")
# # plt.plot(PCA_result[y_hc==8, 0], PCA_result[y_hc==8, 1], "ws")
# # plt.plot(PCA_result[y_hc==9, 0], PCA_result[y_hc==9, 1], "r^")
# # plt.plot(PCA_result[y_hc==10, 0], PCA_result[y_hc==10, 1], "g^")
# # plt.plot(PCA_result[y_hc==11, 0], PCA_result[y_hc==11, 1], "c^")
# # plt.plot(PCA_result[y_hc==12, 0], PCA_result[y_hc==12, 1], "b^")
# # plt.show()

# #dataframe for PCA results and clusters
TSNE_df = pd.DataFrame(TSNE_result, columns=['x_values', 'y_values'])
TSNE_df['word'] = model.wv.index_to_key
TSNE_df['cluster'] = kmeans

# #look at the result of PCA dataframe
# st.write(PCA_df)

# #not sure how time consuming this is but ill do it like this: search for word in dataframe and add it to the cluster value

#split the list of artistGenres from sub_df into lists of individual genres (in order to perform merge with PCA_df)
sub_df['artistGenres'] = sub_df['artistGenres'].str.split(', ')

#explode the list from sub_df into separate rows
sub_df_exploded = sub_df.explode('artistGenres')

# st.write(sub_df_exploded)

#merge the exploded dataframe with PCA_df for the artist genre
df_merged_genre = pd.merge(sub_df_exploded, TSNE_df, left_on='artistGenres', right_on='word', how='left')

#iterate through each line of the cluster and print to a specific csv file
for index, entry in df_merged_genre.iterrows():
    # Check if the 'cluster' value is not NaN
    if pd.notna(entry['cluster']):
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
            unique_genres.add(entry['artistGenres'])

        #print the values from the set to check results
        for genre in unique_genres:
            print(genre)
        
        print()