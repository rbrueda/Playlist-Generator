import pandas as pd


# find unique genres in the playlist results csv
df = pd.read_csv('playlists/playlist_results_4_1.csv')
genres = set()


for index, row in df.iterrows():
    listOfGenres = row['Genres'].split(",")
    for genre in listOfGenres:
        if genre not in genres:
            genres.add(genre)

print("genres in playlist subcluster")
print(genres)


#get the subclusters in subcluster_statistics.csv
df1 = pd.read_csv("playlists/genre_results.csv")

entry = df1.loc[(df1['main_cluster'] == 4) & (df1['sub_cluster'] == 1)]

print(entry['genres'])

for index, row in entry.iterrows():
    listOfGenres2 = row['genres'].split(', ')

genres2 = set()

for genre in listOfGenres:
    genres2.add(genre)

print("genres from subcluster_results")
print(genres2)