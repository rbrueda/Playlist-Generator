#this script is to fine tune a general model from SBERT to specific to music genres
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import Dataset  # Add this line to import Dataset

# Load the pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

train_examples = []

#use the corpus we have created with cosine similarity computed
with open('genre_corpus.txt', 'r') as file:
    for line in file:
        genre1, genre2, similarity_score = line.strip().split(' ')
        train_examples.append(InputExample(texts=[genre1, genre2], label=float(similarity_score)))

#create a dataloader with the training example
train_dataloader = DataLoader(train_examples, shuffle=True)

# Use CosineSimilarityLoss for pairwise cosine similarity learning
train_loss = losses.CosineSimilarityLoss(model=model)

# Fine-tune for a specified number of epochs
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

#save the model
model.save("fine_tuned_genre_sbert")



