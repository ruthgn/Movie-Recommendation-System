*Full project with complete code and datasets available on my Github [repository](https://github.com/ruthgn/Movie-Recommendation-System).*

In recommender systems theory, the two most common types of recommender systems are **Content-Based** and **Collaborative Filtering**. Content-based recommender systems focus on the attributes of items and give you recommendations based on similarities between them. Meanwhile, collaborative filtering produces recommendations based on the knowledge of users' attitude toward items; it uses the "wisdom of the crowd" to produce recommendations. A common example of collaborative filtering is the recommendations Amazon suggests while customers browse for a particular item. Suggestions are based on other shoppers' purchases of similar items and products purchased within the same basket.
The underlying difference between content-based and collaborative filtering lies in how the former focuses on item similarities and the latter, user preferences.

Overall, collaborative filtering is more commonly used in content based systems because it usually gives better results and is relatively easy to understand from an overall implementation perspective. This post outlines one of the approaches I use with collaborative filtering to create a simple movie recommendation system with Python. We will use the famous MovieLens dataset, which is one of the most common datasets used when implementing and testing recommender engines. It contains 100k movie ratings from 943 users and a selection of 1682 movies. You can download the dataset [here](http://files.grouplens.org/datasets/movielens/ml-100k.zip). Full description of the dataset is available [here](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt).  

## Getting Started


```python
# Import the libraries we will need
import numpy as np
import pandas as pd
```

You can download the dataset [here](http://files.grouplens.org/datasets/movielens/ml-100k.zip) or use the **u.data** file that is included in the project Github [repository](https://github.com/ruthgn/Movie-Recommendation-System).


```python
# Create new variable listing columns we want from the dataset
# Specify the separator argument for a Tab separated file when reading data
columnsNames = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=columnsNames)
```


```python
# Take a quick look at the data
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>50</td>
      <td>5</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>172</td>
      <td>5</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>133</td>
      <td>1</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>3</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>4</th>
      <td>186</td>
      <td>302</td>
      <td>3</td>
      <td>891717742</td>
    </tr>
  </tbody>
</table>
</div>



*Note*: Since movie titles are reprented by *item_id* and not the actual movie names, we will use the **Movie_ID_Titles.csv** file I've included in the project [repository](https://github.com/ruthgn/Movie-Recommendation-System) to grab and merge them with the DataFrame we've created.


```python
movieTitles = pd.read_csv('Movie_Id_Titles')
```


```python
# Quick look at Movie_ID_Titles csv file
movieTitles.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GoldenEye (1995)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Four Rooms (1995)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Get Shorty (1995)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Copycat (1995)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge with initial dataframe
df = pd.merge(df, movieTitles, on='item_id')
# Take a look at final dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>50</td>
      <td>5</td>
      <td>881250949</td>
      <td>Star Wars (1977)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>290</td>
      <td>50</td>
      <td>5</td>
      <td>880473582</td>
      <td>Star Wars (1977)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>79</td>
      <td>50</td>
      <td>4</td>
      <td>891271545</td>
      <td>Star Wars (1977)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>50</td>
      <td>5</td>
      <td>888552084</td>
      <td>Star Wars (1977)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>50</td>
      <td>5</td>
      <td>879362124</td>
      <td>Star Wars (1977)</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory Data Analysis

Let's explore the data a bit and take a look at some of the best rated movies!


```python
# Import data visualization libraries and set style
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')
```

We're going to create a ratings DataFrame that contains each movie's average rating score and total count of ratings. But before that, let's look at two groups of movies: one with *high* ratings vs one with *numerous* ratings. Be aware that movies with the highest average ratings in this data set are not necessarily box office hits given the total number of ratings. By contrast, the movie set with the highest total count of ratings does contain box office hits you're likely familiar with.


```python
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
```




    title
    Marlene Dietrich: Shadow and Light (1996)     5.0
    Prefontaine (1997)                            5.0
    Santa with Muscles (1996)                     5.0
    Star Kid (1997)                               5.0
    Someone Else's America (1995)                 5.0
    Name: rating, dtype: float64




```python
df.groupby('title')['rating'].count().sort_values(ascending=False).head()
```




    title
    Star Wars (1977)             584
    Contact (1997)               509
    Fargo (1996)                 508
    Return of the Jedi (1983)    507
    Liar Liar (1997)             485
    Name: rating, dtype: int64




```python
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>2.333333</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>2.600000</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>2.908257</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>4.344000</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>3.024390</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add count of rating column:
ratings['count of rating'] = df.groupby('title')['rating'].count()
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>count of rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>2.333333</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>2.600000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>2.908257</td>
      <td>109</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>4.344000</td>
      <td>125</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>3.024390</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
</div>



Now let's explore the data using some visualizations:


```python
plt.figure(figsize=(11,5))
ratings['rating'].hist(bins=70)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x147167d1a88>




![png](movie-recommendation-system_files/movie-recommendation-system_20_1.png)



```python
plt.figure(figsize=(11,5))
ratings['count of rating'].hist(bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x14717162088>




![png](movie-recommendation-system_files/movie-recommendation-system_21_1.png)


We learned that most movies in the dataset have very few (zero or one) reviews while several others (probably considered 'blockbusters') have reviews numbering in the hundreds.


```python
sns.jointplot(x='rating', y='count of rating', data=ratings)
```




    <seaborn.axisgrid.JointGrid at 0x14717043248>




![png](movie-recommendation-system_files/movie-recommendation-system_23_1.png)


We can also deduce that movies that have been rated by a lot of people tend to be rated positively. It's important to be aware of the fact that the rating scores contained in the dataset are skewed towards mainstream or blockbuster titles; our recommendation system performs better in recommending mainstream or blockbuster movies and for users who have watched and rated more popular titles.

Now that we have a general idea of what the data looks like, let's move on to creating a simple recommendation system:

## Recommending Similar Movies

We will build a recommendation system based on similarities in user perception. Let's create a matrix that has the user IDs on one axis and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. Note that there are a lot of NaN values because most users have not seen most of movies in the dataset.


```python
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
moviemat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>title</th>
      <th>'Til There Was You (1997)</th>
      <th>1-900 (1994)</th>
      <th>101 Dalmatians (1996)</th>
      <th>12 Angry Men (1957)</th>
      <th>187 (1997)</th>
      <th>2 Days in the Valley (1996)</th>
      <th>20,000 Leagues Under the Sea (1954)</th>
      <th>2001: A Space Odyssey (1968)</th>
      <th>3 Ninjas: High Noon At Mega Mountain (1998)</th>
      <th>39 Steps, The (1935)</th>
      <th>...</th>
      <th>Yankee Zulu (1994)</th>
      <th>Year of the Horse (1997)</th>
      <th>You So Crazy (1994)</th>
      <th>Young Frankenstein (1974)</th>
      <th>Young Guns (1988)</th>
      <th>Young Guns II (1990)</th>
      <th>Young Poisoner's Handbook, The (1995)</th>
      <th>Zeus and Roxanne (1997)</th>
      <th>unknown</th>
      <th>Á köldum klaka (Cold Fever) (1994)</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1664 columns</p>
</div>



Most rated movies:


```python
ratings.sort_values('count of rating', ascending=False).head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>count of rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Star Wars (1977)</th>
      <td>4.359589</td>
      <td>584</td>
    </tr>
    <tr>
      <th>Contact (1997)</th>
      <td>3.803536</td>
      <td>509</td>
    </tr>
    <tr>
      <th>Fargo (1996)</th>
      <td>4.155512</td>
      <td>508</td>
    </tr>
    <tr>
      <th>Return of the Jedi (1983)</th>
      <td>4.007890</td>
      <td>507</td>
    </tr>
    <tr>
      <th>Liar Liar (1997)</th>
      <td>3.156701</td>
      <td>485</td>
    </tr>
    <tr>
      <th>English Patient, The (1996)</th>
      <td>3.656965</td>
      <td>481</td>
    </tr>
    <tr>
      <th>Scream (1996)</th>
      <td>3.441423</td>
      <td>478</td>
    </tr>
    <tr>
      <th>Toy Story (1995)</th>
      <td>3.878319</td>
      <td>452</td>
    </tr>
    <tr>
      <th>Air Force One (1997)</th>
      <td>3.631090</td>
      <td>431</td>
    </tr>
    <tr>
      <th>Independence Day (ID4) (1996)</th>
      <td>3.438228</td>
      <td>429</td>
    </tr>
    <tr>
      <th>Raiders of the Lost Ark (1981)</th>
      <td>4.252381</td>
      <td>420</td>
    </tr>
    <tr>
      <th>Godfather, The (1972)</th>
      <td>4.283293</td>
      <td>413</td>
    </tr>
    <tr>
      <th>Pulp Fiction (1994)</th>
      <td>4.060914</td>
      <td>394</td>
    </tr>
    <tr>
      <th>Twelve Monkeys (1995)</th>
      <td>3.798469</td>
      <td>392</td>
    </tr>
    <tr>
      <th>Silence of the Lambs, The (1991)</th>
      <td>4.289744</td>
      <td>390</td>
    </tr>
    <tr>
      <th>Jerry Maguire (1996)</th>
      <td>3.710938</td>
      <td>384</td>
    </tr>
    <tr>
      <th>Chasing Amy (1997)</th>
      <td>3.839050</td>
      <td>379</td>
    </tr>
    <tr>
      <th>Rock, The (1996)</th>
      <td>3.693122</td>
      <td>378</td>
    </tr>
    <tr>
      <th>Empire Strikes Back, The (1980)</th>
      <td>4.206522</td>
      <td>368</td>
    </tr>
    <tr>
      <th>Star Trek: First Contact (1996)</th>
      <td>3.660274</td>
      <td>365</td>
    </tr>
    <tr>
      <th>Titanic (1997)</th>
      <td>4.245714</td>
      <td>350</td>
    </tr>
    <tr>
      <th>Back to the Future (1985)</th>
      <td>3.834286</td>
      <td>350</td>
    </tr>
    <tr>
      <th>Mission: Impossible (1996)</th>
      <td>3.313953</td>
      <td>344</td>
    </tr>
    <tr>
      <th>Fugitive, The (1993)</th>
      <td>4.044643</td>
      <td>336</td>
    </tr>
    <tr>
      <th>Indiana Jones and the Last Crusade (1989)</th>
      <td>3.930514</td>
      <td>331</td>
    </tr>
    <tr>
      <th>Willy Wonka and the Chocolate Factory (1971)</th>
      <td>3.631902</td>
      <td>326</td>
    </tr>
    <tr>
      <th>Princess Bride, The (1987)</th>
      <td>4.172840</td>
      <td>324</td>
    </tr>
    <tr>
      <th>Forrest Gump (1994)</th>
      <td>3.853583</td>
      <td>321</td>
    </tr>
    <tr>
      <th>Saint, The (1997)</th>
      <td>3.123418</td>
      <td>316</td>
    </tr>
    <tr>
      <th>Monty Python and the Holy Grail (1974)</th>
      <td>4.066456</td>
      <td>316</td>
    </tr>
    <tr>
      <th>Full Monty, The (1997)</th>
      <td>3.926984</td>
      <td>315</td>
    </tr>
    <tr>
      <th>Men in Black (1997)</th>
      <td>3.745875</td>
      <td>303</td>
    </tr>
    <tr>
      <th>Terminator, The (1984)</th>
      <td>3.933555</td>
      <td>301</td>
    </tr>
    <tr>
      <th>E.T. the Extra-Terrestrial (1982)</th>
      <td>3.833333</td>
      <td>300</td>
    </tr>
    <tr>
      <th>Dead Man Walking (1995)</th>
      <td>3.896321</td>
      <td>299</td>
    </tr>
    <tr>
      <th>Leaving Las Vegas (1995)</th>
      <td>3.697987</td>
      <td>298</td>
    </tr>
    <tr>
      <th>Schindler's List (1993)</th>
      <td>4.466443</td>
      <td>298</td>
    </tr>
    <tr>
      <th>Braveheart (1995)</th>
      <td>4.151515</td>
      <td>297</td>
    </tr>
    <tr>
      <th>L.A. Confidential (1997)</th>
      <td>4.161616</td>
      <td>297</td>
    </tr>
    <tr>
      <th>Conspiracy Theory (1997)</th>
      <td>3.423729</td>
      <td>295</td>
    </tr>
    <tr>
      <th>Terminator 2: Judgment Day (1991)</th>
      <td>4.006780</td>
      <td>295</td>
    </tr>
    <tr>
      <th>Birdcage, The (1996)</th>
      <td>3.443686</td>
      <td>293</td>
    </tr>
    <tr>
      <th>Twister (1996)</th>
      <td>3.215017</td>
      <td>293</td>
    </tr>
    <tr>
      <th>Mr. Holland's Opus (1995)</th>
      <td>3.778157</td>
      <td>293</td>
    </tr>
    <tr>
      <th>Alien (1979)</th>
      <td>4.034364</td>
      <td>291</td>
    </tr>
    <tr>
      <th>When Harry Met Sally... (1989)</th>
      <td>3.910345</td>
      <td>290</td>
    </tr>
    <tr>
      <th>Aliens (1986)</th>
      <td>3.947183</td>
      <td>284</td>
    </tr>
    <tr>
      <th>Shawshank Redemption, The (1994)</th>
      <td>4.445230</td>
      <td>283</td>
    </tr>
    <tr>
      <th>Jaws (1975)</th>
      <td>3.775000</td>
      <td>280</td>
    </tr>
    <tr>
      <th>Groundhog Day (1993)</th>
      <td>3.764286</td>
      <td>280</td>
    </tr>
  </tbody>
</table>
</div>



I'm going to choose three very different movies to test our recommender system: **Star Wars (1977)**--a sci-fi action, **Toy Story (1995)**--an animated family comedy, and lastly, **The Silence of the Lambs (1991)**--a horror and psychological thriller.


```python
# Grab the user ratings for each movie
starWarsUserRatings = moviemat['Star Wars (1977)']
toyStoryUserRatings = moviemat['Toy Story (1995)']
```


```python
# Use corrwith() method to get correlations between two pandas series
similarToStarWars = moviemat.corrwith(starWarsUserRatings)
similarToToyStory = moviemat.corrwith(toyStoryUserRatings)
```

    C:\Users\Ruth Nainggolan\anaconda3\lib\site-packages\numpy\lib\function_base.py:2526: RuntimeWarning: Degrees of freedom <= 0 for slice
      c = cov(x, y, rowvar)
    C:\Users\Ruth Nainggolan\anaconda3\lib\site-packages\numpy\lib\function_base.py:2455: RuntimeWarning: divide by zero encountered in true_divide
      c *= np.true_divide(1, fact)
    


```python
# Clean the data: Remove NaN values and set as a DataFrame instead of a series
corrStarWars = pd.DataFrame(similarToStarWars, columns=['Correlation'])
corrStarWars.dropna(inplace=True)
corrStarWars.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>0.872872</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>-0.645497</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>0.211132</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>0.184289</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>0.027398</td>
    </tr>
  </tbody>
</table>
</div>



Now, if we sort the DataFrame by correlation, we should get the most similar movies. However, we will get some results that don't really make sense. This is caused by the fact that there are a lot of movies only watched once by users who also watched Star Wars (which is understandable as it was the most popular movie).


```python
corrStarWars.sort_values('Correlation',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Commandments (1997)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Cosi (1996)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>No Escape (1994)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Stripes (1981)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Man of the Year (1995)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Hollow Reed (1996)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Beans of Egypt, Maine, The (1994)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Good Man in Africa, A (1994)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Old Lady Who Walked in the Sea, The (Vieille qui marchait dans la mer, La) (1991)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Outlaw, The (1943)</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



We will fix this by filtering out movies that have less than 100 reviews--this value was chosen based off of what we discovered from the histogram visualization during our exploratory data analysis earlier.


```python
corrStarWars = corrStarWars.join(ratings['count of rating'])
corrStarWars.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
      <th>count of rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>0.872872</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>-0.645497</td>
      <td>5</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>0.211132</td>
      <td>109</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>0.184289</td>
      <td>125</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>0.027398</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sort the values for titles to make a lot more sense
corrStarWars[corrStarWars['count of rating']>100].sort_values('Correlation',
                                                              ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
      <th>count of rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Star Wars (1977)</th>
      <td>1.000000</td>
      <td>584</td>
    </tr>
    <tr>
      <th>Empire Strikes Back, The (1980)</th>
      <td>0.748353</td>
      <td>368</td>
    </tr>
    <tr>
      <th>Return of the Jedi (1983)</th>
      <td>0.672556</td>
      <td>507</td>
    </tr>
    <tr>
      <th>Raiders of the Lost Ark (1981)</th>
      <td>0.536117</td>
      <td>420</td>
    </tr>
    <tr>
      <th>Austin Powers: International Man of Mystery (1997)</th>
      <td>0.377433</td>
      <td>130</td>
    </tr>
  </tbody>
</table>
</div>



These recommendations make a lot of sense! In fact, if you are a Star Wars fan you should definitely be familiar with its sequels, *The Empire Strikes Back* and *Return of the Jedi*, which make up the original Star Wars Trilogy.

Now let's do the same for Toy Story and see what recommendations are in store for us!


```python
corrToyStory = pd.DataFrame(similarToToyStory, columns=['Correlation'])
corrToyStory.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>0.534522</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>0.232118</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>0.334943</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>0.651857</td>
    </tr>
  </tbody>
</table>
</div>




```python
corrToyStory.dropna(inplace=True)
```


```python
corrToyStory = corrToyStory.join(ratings['count of rating'])
```


```python
# Sort the values and display recommendation
corrToyStory[corrToyStory['count of rating']>100].sort_values('Correlation', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
      <th>count of rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toy Story (1995)</th>
      <td>1.000000</td>
      <td>452</td>
    </tr>
    <tr>
      <th>Craft, The (1996)</th>
      <td>0.549100</td>
      <td>104</td>
    </tr>
    <tr>
      <th>Down Periscope (1996)</th>
      <td>0.457995</td>
      <td>101</td>
    </tr>
    <tr>
      <th>Miracle on 34th Street (1994)</th>
      <td>0.456291</td>
      <td>101</td>
    </tr>
    <tr>
      <th>G.I. Jane (1997)</th>
      <td>0.454756</td>
      <td>175</td>
    </tr>
  </tbody>
</table>
</div>


Sure enough, recommendations based off of *Toy Story* are a mix of family-friendly flicks and fun action films.

Lastly, we will check out recommendations for fans of The Silence of the Lambs. 

*Note*: I will be repeating the steps I applied previously to the two movies in the section below. When re-running this project code, you can easily replace variables representing *The Silence of Lambs* with those representing your movie of choice to get a personalized recommendation.


```python
# Grab the user ratings for movie of interest
silenceOfTheLambsUserRatings = moviemat['Silence of the Lambs, The (1991)']
```


```python
# Use corrwith() method to get correlations between two pandas series
similarToSilenceOfTheLambs = moviemat.corrwith(silenceOfTheLambsUserRatings)
```

    C:\Users\Ruth Nainggolan\anaconda3\lib\site-packages\numpy\lib\function_base.py:2526: RuntimeWarning: Degrees of freedom <= 0 for slice
      c = cov(x, y, rowvar)
    C:\Users\Ruth Nainggolan\anaconda3\lib\site-packages\numpy\lib\function_base.py:2455: RuntimeWarning: divide by zero encountered in true_divide
      c *= np.true_divide(1, fact)
    


```python
# Clean the data: Remove NaN values and set as a DataFrame instead of a series
corrSilenceOfTheLambs = pd.DataFrame(similarToSilenceOfTheLambs, columns=['Correlation'])
corrSilenceOfTheLambs.dropna(inplace=True)
corrSilenceOfTheLambs = corrSilenceOfTheLambs.join(ratings['count of rating'])
```


```python
# Sort the values and display recommendations
corrSilenceOfTheLambs[corrSilenceOfTheLambs['count of rating']>100].sort_values('Correlation',
                                                              ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
      <th>count of rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Silence of the Lambs, The (1991)</th>
      <td>1.000000</td>
      <td>390</td>
    </tr>
    <tr>
      <th>Alien: Resurrection (1997)</th>
      <td>0.408675</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Basic Instinct (1992)</th>
      <td>0.403709</td>
      <td>101</td>
    </tr>
    <tr>
      <th>Crying Game, The (1992)</th>
      <td>0.370926</td>
      <td>119</td>
    </tr>
    <tr>
      <th>Shine (1996)</th>
      <td>0.368361</td>
      <td>129</td>
    </tr>
  </tbody>
</table>
</div>



I think the recommendations generated from *The Silence of The Lamb* are particularly great for those who enjoy thrillers and horror titles!

Are you looking for movie recommendations? Feel free to run this code yourself by accessing the project [repository](https://github.com/ruthgn/Movie-Recommendation-System) and inputing your favorite movie!
