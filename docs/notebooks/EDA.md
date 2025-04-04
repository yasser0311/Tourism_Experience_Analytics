# ***Tourism Data - Exploratory Data Analysis***

This notebook performs exploratory data analysis on tourism data including:
 1. User distribution across geographical locations
 2. Attraction type popularity
 3. Visit patterns and demographics
 4. Rating distributions


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display

# Set up visualization style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')
sns.set_palette("husl")
%matplotlib inline

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

```

## **1. Data Loading with Error Handling**


```python
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {os.path.basename(file_path)} successfully")
        return df
    except Exception as e:
        print(f"Error loading {os.path.basename(file_path)}: {e}")
        return None

# Dictionary of dataset paths
data_paths = {
    'user_data': 'D:\\Projects\\Guvi_Project4\\Datasets\\user_data.csv',
    'attraction_data': 'D:\\Projects\\Guvi_Project4\\Datasets\\attraction_type.csv',
    'visit_data': 'D:\\Projects\\Guvi_Project4\\Datasets\\visit_mode.csv',
    'transaction_data': 'D:\\Projects\\Guvi_Project4\\Datasets\\transaction_data.csv', 
    'region_data': 'D:\\Projects\\Guvi_Project4\\Datasets\\region_data.csv',
    'country_data': 'D:\\Projects\\Guvi_Project4\\Datasets\\country_data.csv',
    'continent_data': 'D:\\Projects\\Guvi_Project4\\Datasets\\continent_data.csv',
    'city_data': 'D:\\Projects\\Guvi_Project4\\Datasets\\city_data.csv',
    'item_data': 'D:\\Projects\\Guvi_Project4\\Datasets\\item_data.csv'
}

# Load all datasets
datasets = {name: load_data(path) for name, path in data_paths.items()}
loaded_datasets = {k: v for k, v in datasets.items() if v is not None}

```

    Loaded user_data.csv successfully
    Loaded attraction_type.csv successfully
    Loaded visit_mode.csv successfully
    Loaded transaction_data.csv successfully
    Loaded region_data.csv successfully
    Loaded country_data.csv successfully
    Loaded continent_data.csv successfully
    Loaded city_data.csv successfully
    Loaded item_data.csv successfully
    

## **2. Basic Data Exploration**


```python
def show_basic_stats(df, name):
    """Display basic statistics for a dataframe"""
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 3 rows:")
    display(df.head(3))
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nData types:")
    print(df.dtypes.value_counts())

# Show stats for loaded datasets
for name, df in loaded_datasets.items():
    show_basic_stats(df, name)
```

    
    === user_data ===
    Shape: (33530, 5)
    
    First 3 rows:
    


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
      <th>UserId</th>
      <th>ContinentId</th>
      <th>RegionId</th>
      <th>CountryId</th>
      <th>CityId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14</td>
      <td>5</td>
      <td>20</td>
      <td>155</td>
      <td>220.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>3</td>
      <td>14</td>
      <td>101</td>
      <td>3098.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>4</td>
      <td>15</td>
      <td>109</td>
      <td>4303.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    UserId         0
    ContinentId    0
    RegionId       0
    CountryId      0
    CityId         4
    dtype: int64
    
    Data types:
    int64      4
    float64    1
    Name: count, dtype: int64
    
    === attraction_data ===
    Shape: (17, 2)
    
    First 3 rows:
    


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
      <th>AttractionTypeId</th>
      <th>AttractionType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>Ancient Ruins</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>Ballets</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>Beaches</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    AttractionTypeId    0
    AttractionType      0
    dtype: int64
    
    Data types:
    int64     1
    object    1
    Name: count, dtype: int64
    
    === visit_data ===
    Shape: (6, 2)
    
    First 3 rows:
    


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
      <th>VisitModeId</th>
      <th>VisitMode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Business</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Couples</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    VisitModeId    0
    VisitMode      0
    dtype: int64
    
    Data types:
    int64     1
    object    1
    Name: count, dtype: int64
    
    === transaction_data ===
    Shape: (52930, 7)
    
    First 3 rows:
    


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
      <th>TransactionId</th>
      <th>UserId</th>
      <th>VisitYear</th>
      <th>VisitMonth</th>
      <th>VisitMode</th>
      <th>AttractionId</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>70456</td>
      <td>2022</td>
      <td>10</td>
      <td>2</td>
      <td>640</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>7567</td>
      <td>2022</td>
      <td>10</td>
      <td>4</td>
      <td>640</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>79069</td>
      <td>2022</td>
      <td>10</td>
      <td>3</td>
      <td>640</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    TransactionId    0
    UserId           0
    VisitYear        0
    VisitMonth       0
    VisitMode        0
    AttractionId     0
    Rating           0
    dtype: int64
    
    Data types:
    int64    7
    Name: count, dtype: int64
    
    === region_data ===
    Shape: (22, 3)
    
    First 3 rows:
    


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
      <th>Region</th>
      <th>RegionId</th>
      <th>ContinentId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Central Africa</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>East Africa</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    Region         0
    RegionId       0
    ContinentId    0
    dtype: int64
    
    Data types:
    int64     2
    object    1
    Name: count, dtype: int64
    
    === country_data ===
    Shape: (165, 3)
    
    First 3 rows:
    


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
      <th>CountryId</th>
      <th>Country</th>
      <th>RegionId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Cameroon</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Chad</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    CountryId    0
    Country      0
    RegionId     0
    dtype: int64
    
    Data types:
    int64     2
    object    1
    Name: count, dtype: int64
    
    === continent_data ===
    Shape: (6, 2)
    
    First 3 rows:
    


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
      <th>ContinentId</th>
      <th>Continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>America</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    ContinentId    0
    Continent      0
    dtype: int64
    
    Data types:
    int64     1
    object    1
    Name: count, dtype: int64
    
    === city_data ===
    Shape: (9143, 3)
    
    First 3 rows:
    


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
      <th>CityId</th>
      <th>CityName</th>
      <th>CountryId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Douala</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>South Region</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    Missing values:
    CityId       0
    CityName     1
    CountryId    0
    dtype: int64
    
    Data types:
    int64     2
    object    1
    Name: count, dtype: int64
    

## **3. User Distribution and Geographical Analysis**


```python
def save_plot(fig, filename):
    fig.savefig(f'visualizations/{filename}', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved visualization: {filename}")

if all(k in loaded_datasets for k in ['user_data', 'city_data', 'region_data', 'country_data', 'continent_data']):
    try:
        # Verify column names in each dataset
        print("\nActual column names:")
        print("country_data:", loaded_datasets['country_data'].columns.tolist())
        print("continent_data:", loaded_datasets['continent_data'].columns.tolist())
        print("region_data:", loaded_datasets['region_data'].columns.tolist())
        print("city_data:", loaded_datasets['city_data'].columns.tolist())
        print("user_data:", loaded_datasets['user_data'].columns.tolist())
        
        # Based on your error output, let's adjust the merge logic:
        # 1. First merge region with continent (since region has ContinentId)
        region_continent = pd.merge(
            loaded_datasets['region_data'],
            loaded_datasets['continent_data'],
            on='ContinentId',
            how='left'
        )
        
        # 2. Then merge country with the above result
        country_merged = pd.merge(
            loaded_datasets['country_data'],
            region_continent,
            on='RegionId',
            how='left'
        )
        
        # 3. Then merge city data
        geo_data = pd.merge(
            loaded_datasets['city_data'],
            country_merged,
            on='CountryId',
            how='left'
        )
        
        # 4. Finally merge with user data
        user_geo = pd.merge(
            loaded_datasets['user_data'],
            geo_data,
            on='CityId',
            how='left'
        )
        
        # Clean up column names
        user_geo.columns = [col.split('_')[0] if '_' in col else col for col in user_geo.columns]
        
        # Fill missing values
        user_geo['Continent'] = user_geo['Continent'].fillna('Unknown')
        user_geo['Country'] = user_geo['Country'].fillna('Unknown')
        
        # Plot continent distribution
        plt.figure(figsize=(12, 6))
        continent_dist = user_geo[user_geo['Continent'] != 'Unknown']['Continent'].value_counts().sort_values(ascending=False)
        
        ax = sns.barplot(
            x=continent_dist.index,
            y=continent_dist.values,
            order=continent_dist.index
        )
        plt.title('User Distribution by Continent')
        plt.xticks(rotation=45)
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():,.0f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', 
                       xytext=(0, 10), 
                       textcoords='offset points')
        
        save_plot(plt.gcf(), 'user_distribution_by_continent.png')
        
        # Plot top countries
        plt.figure(figsize=(12, 6))
        country_dist = user_geo[user_geo['Country'] != 'Unknown']['Country'].value_counts().nlargest(10).sort_values()
        country_dist.plot(kind='barh')
        plt.title('Top 10 Countries by User Count')
        
        # Add value labels
        for i, v in enumerate(country_dist.values):
            plt.text(v + 3, i, str(v), color='black', va='center')
        
        save_plot(plt.gcf(), 'top_countries_by_users.png')
        
        print("\nSuccessfully generated geographic visualizations")
        
    except Exception as e:
        print(f"\nError during geographic analysis: {str(e)}")
        print("Debug Info:")
        try:
            print("\nAfter region-continent merge:", region_continent.columns.tolist())
            print("After country merge:", country_merged.columns.tolist())
            print("After city merge:", geo_data.columns.tolist())
        except:
            pass
else:
    print("Insufficient data for geographic analysis - missing required datasets")
```

    
    Actual column names:
    country_data: ['CountryId', 'Country', 'RegionId']
    continent_data: ['ContinentId', 'Continent']
    region_data: ['Region', 'RegionId', 'ContinentId']
    city_data: ['CityId', 'CityName', 'CountryId']
    user_data: ['UserId', 'ContinentId', 'RegionId', 'CountryId', 'CityId']
    Saved visualization: user_distribution_by_continent.png
    Saved visualization: top_countries_by_users.png
    
    Successfully generated geographic visualizations
    

## **4. Attraction Analysis**


```python
if all(k in loaded_datasets for k in ['transaction_data', 'attraction_data', 'visit_data', 'item_data']):
    try:
        # Merge attraction data through item_data
        trans_merged = pd.merge(
            loaded_datasets['transaction_data'],
            pd.merge(
                loaded_datasets['item_data'],
                loaded_datasets['attraction_data'],
                on='AttractionTypeId'
            ),
            on='AttractionId'
        )
        
        # Merge with visit data
        trans_merged = pd.merge(
            trans_merged,
            loaded_datasets['visit_data'],
            left_on='VisitMode',
            right_on='VisitModeId',
            how='left'
        )
        
        # Top attractions by visits
        plt.figure(figsize=(14, 6))
        top_attractions = trans_merged['AttractionType'].value_counts().nlargest(10)
        ax = sns.barplot(x=top_attractions.values, y=top_attractions.index, orient='h')
        plt.title('Top 10 Most Visited Attraction Types')
        plt.xlabel('Number of Visits')
        save_plot(plt.gcf(), 'top_attractions_by_visits.png')
        
        # Top attractions by rating
        plt.figure(figsize=(14, 6))
        avg_ratings = trans_merged.groupby('AttractionType')['Rating'].mean().nlargest(10)
        ax = sns.barplot(x=avg_ratings.values, y=avg_ratings.index, orient='h')
        plt.title('Top 10 Highest Rated Attraction Types')
        plt.xlabel('Average Rating')
        save_plot(plt.gcf(), 'top_attractions_by_rating.png')
        
    except Exception as e:
        print(f"Attraction analysis error: {e}")
else:
    print("Insufficient data for attraction analysis")

```

    Saved visualization: top_attractions_by_visits.png
    Saved visualization: top_attractions_by_rating.png
    

## **5. Rating Analysis**


```python
if 'transaction_data' in loaded_datasets and 'Rating' in loaded_datasets['transaction_data'].columns:
    try:
        # Rating distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(loaded_datasets['transaction_data']['Rating'], bins=10, kde=True)
        plt.title('Distribution of Ratings')
        save_plot(plt.gcf(), 'rating_distribution.png')
        
        # Monthly rating trends
        if 'VisitYear' in loaded_datasets['transaction_data'].columns and 'VisitMonth' in loaded_datasets['transaction_data'].columns:
            trans_data = loaded_datasets['transaction_data'].copy()
            trans_data['YearMonth'] = trans_data['VisitYear'].astype(str) + '-' + trans_data['VisitMonth'].astype(str).str.zfill(2)
            monthly_ratings = trans_data.groupby('YearMonth')['Rating'].mean()
            
            plt.figure(figsize=(14, 6))
            monthly_ratings.plot(marker='o')
            plt.title('Monthly Average Ratings')
            plt.xlabel('Month')
            plt.ylabel('Average Rating')
            plt.grid(True)
            save_plot(plt.gcf(), 'monthly_rating_trends.png')
            
    except Exception as e:
        print(f"Rating analysis error: {e}")
else:
    print("No rating data available")

```

    Saved visualization: rating_distribution.png
    Saved visualization: monthly_rating_trends.png
    

## **6. Visit Mode Analysis**


```python
if 'visit_data' in loaded_datasets:
    try:
        # Visit mode distribution
        plt.figure(figsize=(10, 6))
        visit_counts = loaded_datasets['visit_data']['VisitMode'].value_counts()
        sns.barplot(x=visit_counts.index, y=visit_counts.values)
        plt.title('Visit Mode Distribution')
        plt.xticks(rotation=45)
        save_plot(plt.gcf(), 'visit_mode_distribution.png')
        
    except Exception as e:
        print(f"Visit mode analysis error: {e}")
else:
    print("No visit data available")

```

    Saved visualization: visit_mode_distribution.png
    

## **7. Summary**


```python
print("\n=== EDA Complete ===")
print(f"Saved visualizations to {os.path.abspath('visualizations')}")
print("\nGenerated visualizations:")
for viz_file in os.listdir('visualizations'):
    if viz_file.endswith('.png'):
        print(f"- {viz_file}")
```

    
    === EDA Complete ===
    Saved visualizations to d:\Projects\Guvi_Project4\visualizations
    
    Generated visualizations:
    - monthly_rating_trends.png
    - rating_distribution.png
    - top_attractions_by_rating.png
    - top_attractions_by_visits.png
    - top_countries_by_users.png
    - user_distribution_by_continent.png
    - visit_mode_distribution.png
    
