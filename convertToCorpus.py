#this script is to convert the music genres json file to a text file used as a corpus for our fasttext model
import json
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util #util often yields better results
import numpy as np

with open('music_genres.json') as json_file:
  jsonObject = json.load(json_file)

#extract all the 'name' attribute values in the JSON object
genreNames = [(item['name'].lower()).replace(' ', '_') for item in jsonObject]
print(genreNames)

#load a SBERT model and generate embeddings for each genre in json file
model = SentenceTransformer('all-MiniLM-L6-v2') #use this one for broader semantic model
genre_embeddings = model.encode(genreNames)

#generate all possible pairs of names
genre_pairs = list(itertools.combinations(enumerate(genreNames), 2))

#generate a similarity threshold
threshold = 0.80 #more specific
similarity_pairs = []

numberOfCombinations = 0
numberOfSimilarities = 0

#go through each pair and compute the cosine simiarity between each
for (i, genre1), (j, genre2) in genre_pairs:
  numberOfCombinations += 1
  similarity_score = util.cos_sim(genre_embeddings[i], genre_embeddings[j]).item()

  similarity_score = np.clip(similarity_score, -1, 1) #music be in this range
  similarity_pairs.append((genre1, genre2, similarity_score))


  if similarity_score >= threshold:
    numberOfSimilarities += 1

print(f"number of combinations: {numberOfCombinations}")
print(f"number of similarities: {numberOfSimilarities}")
#calculate reduction percent -- seems like dataset is too broad or threshold is too low
reductionPercent = ((numberOfCombinations-numberOfSimilarities)/numberOfCombinations)*100
print("-----------------------------------------------------------")
print(f"Reduction of combinations: {reductionPercent}")


#create to corpus for fine-tuning FastText model
with open('genre_corpus.txt', 'w') as corpus_file:
  for genre1, genre2, score in similarity_pairs:
    corpus_file.write(f"{genre1} {genre2} {score}\n")




