import pandas as pd

df = pd.read_csv('sampleSongs.csv')

#* albumType -> maybe? (options: single or album)

#here will list the columns to find relevant information from 
metricColumns = ['artistName', 'albumPopularity', 'albumReleaseDate', 'artistGenres']

#make a new dataframe with just these columns
sub_df = df[metricColumns]


print(sub_df.head())