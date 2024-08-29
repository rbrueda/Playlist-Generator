
#todo: find the dimension length from the data -- find the most optimal scale number using a mathematical equation
#todo: figure out the most optimal min_count number -- this value probably works best with 1, as we want all values to produce some unique value -- but im not sure how it will behave with "dance pop" versus "pop" -- similar words

#todo: figure out issue and propose solution 
#todo: goal - finish this part tonight so i can move onto different playlist cateogries --> popularity, release dates, artists name (should be simular to word to vec? -- but i think it isnt really -- we need to find the most popular)

import pandas as pd
import re
import numpy as np

import nltk
from nltk.corpus import stopwords

# import word-2-vec model
from gensim.models import Word2Vec

#for visualizing data
import streamlit as st
import altair as alt

#for clustering the data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#function to create a collection of data
def build_collection(data):
    collection = [] #collection of words
    counter = 0 
    
    for index, sentence in data.items():
        if pd.notna(sentence):
            print(sentence)
            counter += 1
            word_list = sentence.split(", ")  #split the list of words by separate names
            collection.append(word_list)

    print(collection)

    return collection


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

#train a word2vec model
model = Word2Vec(collection_of_words, vector_size=20, min_count=1)

#create a keyed vector instance
keyed_vectors = model.wv

#vectorize the list of words
vectors = keyed_vectors[model.wv.index_to_key]
words = list(vectors)

#creates an instance of a 2-d PCA
pca = PCA(n_components=2)

#fit the vectors in the PCA object created -- used for dimensionality reduction
PCA_result = pca.fit_transform(vectors)


#this is what we will start with for the number of clusters -- the number of broad categories
genres = ["Hip Hop", "Pop", "Rock", "R&B", "Rap", "Country", "K-Pop", "Lo-fi", "Indie", "Metal", "Punk", "Jazz", "Classical"]

kmeans = KMeans(n_clusters=len(genres))

#fit K-means model with k=5
kmeans.fit(PCA_result)

#algorithm used for parititioning a dataset into a pre-defined number of clusters
clusters = kmeans.predict(PCA_result)

#dataframe for PCA results and clusters
PCA_df = pd.DataFrame(PCA_result, columns=['x_values', 'y_values'])
PCA_df['word'] = model.wv.index_to_key
PCA_df['cluster'] = clusters

#look at the result of PCA dataframe
st.write(PCA_df)

#not sure how time consuming this is but ill do it like this: search for word in dataframe and add it to the cluster value

#split the list of artistGenres from sub_df into lists of individual genres (in order to perform merge with PCA_df)
sub_df['artistGenres'] = sub_df['artistGenres'].str.split(', ')

#explode the list from sub_df into separate rows
sub_df_exploded = sub_df.explode('artistGenres')

st.write(sub_df_exploded)

#merge the exploded dataframe with PCA_df for the artist genre
df_merged_genre = pd.merge(sub_df_exploded, PCA_df, left_on='artistGenres', right_on='word', how='left')

st.write(df_merged_genre)

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

