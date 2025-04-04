# ***Recommendations: Personalized Attraction Suggestions***

This notebook implements:
 1. Collaborative Filtering (User-Item Matrix)
 2. Content-Based Filtering
 3. Hybrid Recommendation Approach

## **1. Data Preparation**


```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Load datasets
print("Loading datasets...")
try:
    ratings = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\transaction_data.csv')
    attractions = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\attraction_type.csv')
    items = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\item_data.csv')
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    raise

# Merge attraction data
attraction_data = pd.merge(items, attractions, on='AttractionTypeId')

# Verify required columns
print("\nChecking required columns...")
required_cols = {'ratings': ['UserId', 'AttractionId', 'Rating'],
                'attraction_data': ['AttractionId', 'AttractionType', 'AttractionAddress']}

for df_name, cols in required_cols.items():
    missing = set(cols) - set(eval(df_name).columns)
    if missing:
        raise ValueError(f"Missing columns in {df_name}: {missing}")

# Prepare rating matrix
print("\nPreparing rating matrix...")
ratings = ratings[['UserId', 'AttractionId', 'Rating']].dropna()

# Filter users with at least 3 ratings
user_counts = ratings['UserId'].value_counts()
valid_users = user_counts[user_counts >= 3].index
ratings = ratings[ratings['UserId'].isin(valid_users)]

```

    Loading datasets...
    
    Checking required columns...
    
    Preparing rating matrix...
    

## **2. Collaborative Filtering**

### **2.1 User-Item Matrix**


```python
# Create user-item matrix
user_item_matrix = ratings.pivot_table(
    index='UserId',
    columns='AttractionId',
    values='Rating',
    fill_value=0
)

# Convert to sparse matrix
sparse_matrix = csr_matrix(user_item_matrix.values)

# Calculate cosine similarity
user_similarity = cosine_similarity(sparse_matrix)
item_similarity = cosine_similarity(sparse_matrix.T)

# Convert to DataFrames
user_sim_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

item_sim_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)
```

### **2.2 Recommendation Function**


```python
def collaborative_recommend(user_id, n=5):
    """
    Get top N attraction recommendations for a user using collaborative filtering
    """
    if user_id not in user_sim_df.index:
        return pd.DataFrame(columns=attraction_data.columns)
    
    # Get similar users
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:6]
    
    # Get attractions rated by similar users
    similar_users_ratings = ratings[ratings['UserId'].isin(similar_users.index)]
    
    # Exclude attractions already rated by target user
    user_rated = ratings[ratings['UserId'] == user_id]['AttractionId']
    recommendations = similar_users_ratings[~similar_users_ratings['AttractionId'].isin(user_rated)]
    
    # Get top rated attractions by similar users
    if recommendations.empty:
        return pd.DataFrame(columns=attraction_data.columns)
    
    top_attractions = recommendations.groupby('AttractionId')['Rating'].mean()
    top_attractions = top_attractions.sort_values(ascending=False).head(n)
    
    return attraction_data[attraction_data['AttractionId'].isin(top_attractions.index)]

# Test recommendation
print("\nTesting collaborative filtering...")
if not ratings.empty:
    sample_user = ratings['UserId'].sample(1).values[0]
    print(f"\nRecommendations for user {sample_user}:")
    display(collaborative_recommend(sample_user))
else:
    print("No valid ratings data available")
```

    
    Testing collaborative filtering...
    
    Recommendations for user 66324:
    


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
      <th>AttractionId</th>
      <th>AttractionCityId</th>
      <th>AttractionTypeId</th>
      <th>Attraction</th>
      <th>AttractionAddress</th>
      <th>AttractionType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>640</td>
      <td>1</td>
      <td>63</td>
      <td>Sacred Monkey Forest Sanctuary</td>
      <td>Jl. Monkey Forest, Ubud 80571 Indonesia</td>
      <td>Nature &amp; Wildlife Areas</td>
    </tr>
  </tbody>
</table>
</div>


## **3. Content-Based Filtering**

### **3.1 Feature Engineering**


```python
# Prepare attraction features
print("\nPreparing content-based features...")
attraction_data['Features'] = (
    attraction_data['AttractionType'].fillna('') + " " +
    attraction_data['AttractionAddress'].fillna('')
)

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(attraction_data['Features'])

# Calculate cosine similarity
content_sim = cosine_similarity(tfidf_matrix)

# Create similarity DataFrame
content_sim_df = pd.DataFrame(
    content_sim,
    index=attraction_data['AttractionId'],
    columns=attraction_data['AttractionId']
)

```

    
    Preparing content-based features...
    

### **3.2 Recommendation Function**


```python
def content_based_recommend(attraction_id, n=5):
    """
    Get top N similar attractions based on content
    """
    if attraction_id not in content_sim_df.columns:
        return pd.DataFrame(columns=attraction_data.columns)
    
    similar_attractions = content_sim_df[attraction_id].sort_values(ascending=False)[1:n+1]
    return attraction_data[attraction_data['AttractionId'].isin(similar_attractions.index)]

# Test recommendation
print("\nTesting content-based filtering...")
if not attraction_data.empty:
    sample_attraction = attraction_data['AttractionId'].sample(1).values[0]
    print(f"\nAttractions similar to {sample_attraction}:")
    display(content_based_recommend(sample_attraction))
else:
    print("No attraction data available")

```

    
    Testing content-based filtering...
    
    Attractions similar to 824:
    


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
      <th>AttractionId</th>
      <th>AttractionCityId</th>
      <th>AttractionTypeId</th>
      <th>Attraction</th>
      <th>AttractionAddress</th>
      <th>AttractionType</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>737</td>
      <td>1</td>
      <td>76</td>
      <td>Tanah Lot Temple</td>
      <td>Kecamatan Kediri, Kabupaten Tabanan, Beraban 8...</td>
      <td>Religious Sites</td>
      <td>Religious Sites Kecamatan Kediri, Kabupaten Ta...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>748</td>
      <td>1</td>
      <td>72</td>
      <td>Tegalalang Rice Terrace</td>
      <td>Jalan Raya Ceking, Tegalalang 80517 Indonesia</td>
      <td>Points of Interest &amp; Landmarks</td>
      <td>Points of Interest &amp; Landmarks Jalan Raya Ceki...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>749</td>
      <td>1</td>
      <td>93</td>
      <td>Tegenungan Waterfall</td>
      <td>Jl. Raya Tegenungan, Kemenuh, Ubud 80581 Indon...</td>
      <td>Waterfalls</td>
      <td>Waterfalls Jl. Raya Tegenungan, Kemenuh, Ubud ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>841</td>
      <td>1</td>
      <td>92</td>
      <td>Waterbom Bali</td>
      <td>Jl. Kartika Plaza, Kuta 80361 Indonesia</td>
      <td>Water Parks</td>
      <td>Water Parks Jl. Kartika Plaza, Kuta 80361 Indo...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1297</td>
      <td>3</td>
      <td>44</td>
      <td>Yogyakarta Palace</td>
      <td>Yogyakarta</td>
      <td>Historic Sites</td>
      <td>Historic Sites Yogyakarta</td>
    </tr>
  </tbody>
</table>
</div>


## **4. Hybrid Recommendation System**


```python
def hybrid_recommend(user_id, n=5):
    """
    Combine collaborative and content-based filtering
    """
    # Get collaborative recommendations
    collab_recs = collaborative_recommend(user_id, n*2)
    
    if collab_recs.empty:
        return pd.DataFrame(columns=attraction_data.columns)
    
    # Get content-based recommendations for each collab recommendation
    hybrid_recs = pd.DataFrame()
    
    for _, row in collab_recs.iterrows():
        content_recs = content_based_recommend(row['AttractionId'], 2)
        hybrid_recs = pd.concat([hybrid_recs, content_recs])
    
    # Remove duplicates and sort
    if not hybrid_recs.empty:
        hybrid_recs = hybrid_recs.drop_duplicates(subset=['AttractionId'])
        if 'AttractionId' in hybrid_recs.columns:
            hybrid_recs = hybrid_recs.sort_values(by='AttractionId').head(n)
    
    return hybrid_recs

# Test hybrid recommendation
print("\nTesting hybrid recommendation...")
if not ratings.empty and not attraction_data.empty:
    sample_user = ratings['UserId'].sample(1).values[0]
    print(f"\nHybrid recommendations for user {sample_user}:")
    display(hybrid_recommend(sample_user))
else:
    print("Insufficient data for hybrid recommendations")

```

    
    Testing hybrid recommendation...
    
    Hybrid recommendations for user 59746:
    


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
      <th>AttractionId</th>
      <th>AttractionCityId</th>
      <th>AttractionTypeId</th>
      <th>Attraction</th>
      <th>AttractionAddress</th>
      <th>AttractionType</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


## **5. Saving Recommendation Models***


```python
import joblib
import os

# Create directory if it doesn't exist
os.makedirs('tourism_models/recommendation', exist_ok=True)

# Save components
print("\nSaving recommendation models...")
try:
    joblib.dump(user_sim_df, 'tourism_models/recommendation/user_similarity.pkl')
    joblib.dump(item_sim_df, 'tourism_models/recommendation/item_similarity.pkl')
    joblib.dump(content_sim_df, 'tourism_models/recommendation/content_similarity.pkl')
    joblib.dump(tfidf, 'tourism_models/recommendation/tfidf_vectorizer.pkl')
    attraction_data.to_pickle('tourism_models/recommendation/attraction_data.pkl')
    print("All components saved successfully!")
except Exception as e:
    print(f"Error saving models: {e}")

```

    
    Saving recommendation models...
    All components saved successfully!
    

## **6. Production Recommendation Function**


```python
def get_recommendations(user_id=None, attraction_id=None, n=5, method='hybrid'):
    """
    Unified recommendation function for production use
    
    Parameters:
    - user_id: For collaborative/hybrid recommendations
    - attraction_id: For content-based recommendations
    - n: Number of recommendations
    - method: 'collaborative', 'content', or 'hybrid'
    """
    try:
        if method == 'collaborative' and user_id is not None:
            return collaborative_recommend(user_id, n)
        elif method == 'content' and attraction_id is not None:
            return content_based_recommend(attraction_id, n)
        elif method == 'hybrid' and user_id is not None:
            return hybrid_recommend(user_id, n)
        else:
            print("Invalid parameters for recommendation method")
            return pd.DataFrame(columns=attraction_data.columns)
    except Exception as e:
        print(f"Recommendation error: {e}")
        return pd.DataFrame(columns=attraction_data.columns)

# Example usage
print("\nProduction recommendation examples:")
if not ratings.empty and not attraction_data.empty:
    sample_user = ratings['UserId'].sample(1).values[0]
    sample_attraction = attraction_data['AttractionId'].sample(1).values[0]
    
    print("\nCollaborative:")
    display(get_recommendations(user_id=sample_user, method='collaborative'))
    
    print("\nContent-Based:")
    display(get_recommendations(attraction_id=sample_attraction, method='content'))
    
    print("\nHybrid:")
    display(get_recommendations(user_id=sample_user, method='hybrid'))
else:
    print("Insufficient data for demonstration")
```

    
    Production recommendation examples:
    
    Collaborative:
    


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
      <th>AttractionId</th>
      <th>AttractionCityId</th>
      <th>AttractionTypeId</th>
      <th>Attraction</th>
      <th>AttractionAddress</th>
      <th>AttractionType</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>369</td>
      <td>1</td>
      <td>13</td>
      <td>Kuta Beach - Bali</td>
      <td>Kuta</td>
      <td>Beaches</td>
      <td>Beaches Kuta</td>
    </tr>
    <tr>
      <th>5</th>
      <td>737</td>
      <td>1</td>
      <td>76</td>
      <td>Tanah Lot Temple</td>
      <td>Kecamatan Kediri, Kabupaten Tabanan, Beraban 8...</td>
      <td>Religious Sites</td>
      <td>Religious Sites Kecamatan Kediri, Kabupaten Ta...</td>
    </tr>
  </tbody>
</table>
</div>


    
    Content-Based:
    


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
      <th>AttractionId</th>
      <th>AttractionCityId</th>
      <th>AttractionTypeId</th>
      <th>Attraction</th>
      <th>AttractionAddress</th>
      <th>AttractionType</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>640</td>
      <td>1</td>
      <td>63</td>
      <td>Sacred Monkey Forest Sanctuary</td>
      <td>Jl. Monkey Forest, Ubud 80571 Indonesia</td>
      <td>Nature &amp; Wildlife Areas</td>
      <td>Nature &amp; Wildlife Areas Jl. Monkey Forest, Ubu...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>748</td>
      <td>1</td>
      <td>72</td>
      <td>Tegalalang Rice Terrace</td>
      <td>Jalan Raya Ceking, Tegalalang 80517 Indonesia</td>
      <td>Points of Interest &amp; Landmarks</td>
      <td>Points of Interest &amp; Landmarks Jalan Raya Ceki...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>824</td>
      <td>1</td>
      <td>76</td>
      <td>Uluwatu Temple</td>
      <td>Jl. Raya Uluwatu Southern part of Bali, Pecatu...</td>
      <td>Religious Sites</td>
      <td>Religious Sites Jl. Raya Uluwatu Southern part...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>897</td>
      <td>2</td>
      <td>93</td>
      <td>Coban Rondo Waterfall</td>
      <td>Malang District</td>
      <td>Waterfalls</td>
      <td>Waterfalls Malang District</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1280</td>
      <td>3</td>
      <td>72</td>
      <td>Water Castle (Tamansari)</td>
      <td>Jl. Taman, 55133 Indonesia</td>
      <td>Points of Interest &amp; Landmarks</td>
      <td>Points of Interest &amp; Landmarks Jl. Taman, 5513...</td>
    </tr>
  </tbody>
</table>
</div>


    
    Hybrid:
    


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
      <th>AttractionId</th>
      <th>AttractionCityId</th>
      <th>AttractionTypeId</th>
      <th>Attraction</th>
      <th>AttractionAddress</th>
      <th>AttractionType</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>650</td>
      <td>1</td>
      <td>13</td>
      <td>Sanur Beach</td>
      <td>Sanur</td>
      <td>Beaches</td>
      <td>Beaches Sanur</td>
    </tr>
    <tr>
      <th>4</th>
      <td>673</td>
      <td>1</td>
      <td>13</td>
      <td>Seminyak Beach</td>
      <td>Seminyak</td>
      <td>Beaches</td>
      <td>Beaches Seminyak</td>
    </tr>
    <tr>
      <th>8</th>
      <td>824</td>
      <td>1</td>
      <td>76</td>
      <td>Uluwatu Temple</td>
      <td>Jl. Raya Uluwatu Southern part of Bali, Pecatu...</td>
      <td>Religious Sites</td>
      <td>Religious Sites Jl. Raya Uluwatu Southern part...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1297</td>
      <td>3</td>
      <td>44</td>
      <td>Yogyakarta Palace</td>
      <td>Yogyakarta</td>
      <td>Historic Sites</td>
      <td>Historic Sites Yogyakarta</td>
    </tr>
  </tbody>
</table>
</div>

