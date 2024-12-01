# Playlist Generator

## Current Process
1. Get list of unique genres from Liked Songs playlist.

2. Ran a sentence tranformer (SBERT for semantic similarity) on a fine tuned model, fine tuned over a corpus of over 950 000 genre pair combinations using cosine similarity.

3. Got the genre vector embeddings and peformed pca (for dimension reduction), and umap (for adding high granualinty to clusters).

4. Ran HBDSCAN clustering algorithm over umap results to perform clustering. This allows for handling aritrary shape, since the types and number of genres can vastly change depending on users Liked Songs playlist from spotify.

5. Ran fine tuning on clustering by checking every cluster to see if subclustering needs to be performed. Subclustering would need to be performed based on if average distance to centroid exceeds 3 or length of genres per cluster exceeds 20. 

6. Ran a generate playlist function that would go through the all the clusters made and remove clusters whose newest recently played song is from over a year ago (user does not have interest in these group of songs most likely anymore). 

7. Ran a transformer model (gpt 4) over the list of genres for that cluster to label the playlist (for example if the genres in the cluster are pop, dance_pop, latina_pop, k_pop, it would generalize the playlist with a title "Because you most listen to "pop" music").

8. Each cluster also goes through another filtering stage that will check if the songs fit the features of a typical "genre" category. It will check if songs fit the qualities of liveness, acousticness, and tempo.

9. Once each cluster has been labeled and filtered, will check which clusters will make meaningful playlists by finding playlists with majority songs.

10. Generate playlists for top 3 liked artists in Liked Songs album. 

11. Save all these playlists as CSV results 

## Findings 
- Due to the limitation on the number of genres, I have discovered the genres are vectorized in a low dimension space. This is where I applied pca for dimension reduction and spread of 5 on umap. Adding these modifications allows for mre clear cluster results.

![image](https://github.com/user-attachments/assets/7e0d79c2-d20c-4492-95c4-0bb596a4905f)
![image](https://github.com/user-attachments/assets/2199af61-ea12-4587-8d4e-96eea09f7fdb)

