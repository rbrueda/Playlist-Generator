import pandas as pd
from nltk.corpus import stopwords
import re
import nltk
import numpy

# import word-2-vec model
from gensim.models import Word2Vec

from sklearn.decomposition import PCA


# #function to preprocess the data
# def preprocess(text):
#     #remove any characters that are not alphanumeric
#     text_input = re.sub('[^a-zA-Z1-9&,]+', ' ', str(text))
#     output = re.sub(r'\d+', '',text_input)
#     return output.lower().strip()

# #function to remove stopword in the data
# def remove_stopwords(text):
#     filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
#     return " ".join(filtered_words)

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

#* albumType -> maybe? (options: single or album)

#here will list the columns to find relevant information from 
metricColumns = ['artistName', 'albumPopularity', 'albumReleaseDate', 'artistGenres']

#make a new dataframe with just these columns
sub_df = df[metricColumns]

nltk.download('stopwords')

#gets all the stopwords in english
stop = set(stopwords.words("english"))

# sub_df['artistGenres'] = sub_df.artistGenres.map(preprocess)
# sub_df['artistGenres'] = sub_df.artistGenres.map(remove_stopwords)

#create a collection of words from album genres
collection_of_words = build_collection(sub_df['artistGenres'])

#todo: find the dimension length from the data -- find the most optimal scale number using a mathematical equation
#todo: figure out the most optimal min_count number -- this value probably works best with 1, as we want all values to produce some unique value -- but im not sure how it will behave with "dance pop" versus "pop" -- similar words

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

#create a dataframe for the list of vectors
words = pd.DataFrame(words)

#convert PCA to dataframe format
PCA_result = pd.DataFrame(PCA_result)

PCA_result['x_values'] = PCA_result.iloc[0:, 0]
PCA_result['y_values'] = PCA_result.iloc[0:, 1]

PCA_final = pd.merge(words, PCA_result, left_index=True, right_index=True)
PCA_final['word'] = PCA_final.iloc[0:, 0]

PCA_data_complet = PCA_final[['word', 'x_values', 'y_values']]

print(PCA_data_complet)

#todo: plot data on streamlit and look at behavior --- may need to scale the data