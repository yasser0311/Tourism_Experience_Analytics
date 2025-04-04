# ***Tourism Experience Analytics - Data Preprocessing and Feature Engineering***

This notebook handles:
 1. Data loading and initial exploration
 2. Data cleaning and standardization
 3. Feature engineering
 4. Final dataset preparation for modeling

## **1. Data Loading and Initial Exploration**


```python
import pandas as pd
import matplotlib.pyplot as plt

# Load all datasets
print("Loading datasets...")
user_data = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\user_data.csv')
attraction_data = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\attraction_type.csv')
visit_data = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\visit_mode.csv')
transaction_data = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\transaction_data.csv')
region_data = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\region_data.csv')
item_data = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\item_data.csv')
country_data = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\country_data.csv')
continent_data = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\continent_data.csv')
city_data = pd.read_csv('D:\\Projects\\Guvi_Project4\\Datasets\\city_data.csv')

print("\nData loaded successfully!")
print("User data shape:", user_data.shape)
print("Attraction data shape:", attraction_data.shape)   
print("Visit data shape:", visit_data.shape)
print("Transaction data shape:", transaction_data.shape)
print("Region data shape:", region_data.shape)
print("Item data shape:", item_data.shape)
print("Country data shape:", country_data.shape)
print("Continent data shape:", continent_data.shape)
print("City data shape:", city_data.shape)
```

    Loading datasets...
    
    Data loaded successfully!
    User data shape: (33530, 5)
    Attraction data shape: (17, 2)
    Visit data shape: (6, 2)
    Transaction data shape: (52930, 7)
    Region data shape: (22, 3)
    Item data shape: (30, 5)
    Country data shape: (165, 3)
    Continent data shape: (6, 2)
    City data shape: (9143, 3)
    

### **Initial Data Quality Check**


```python
def check_data_quality(df, name):
    """Helper function to check data quality metrics"""
    print(f"\n=== {name} Data Quality ===")
    print(f"Shape: {df.shape}")
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nData types:")
    print(df.dtypes.value_counts())
    if 'Rating' in df.columns:
        print("\nRating stats:")
        print(df['Rating'].describe())
    if 'VisitDate' in df.columns:
        print("\nDate range:")
        print(df['VisitDate'].min(), "to", df['VisitDate'].max())

check_data_quality(user_data, "User")
check_data_quality(city_data, "City")
check_data_quality(transaction_data, "Transaction")
check_data_quality(visit_data, "Visit")
check_data_quality(attraction_data, "Attraction")

```

    
    === User Data Quality ===
    Shape: (33530, 5)
    
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
    
    === City Data Quality ===
    Shape: (9143, 3)
    
    Missing values:
    CityId       0
    CityName     1
    CountryId    0
    dtype: int64
    
    Data types:
    int64     2
    object    1
    Name: count, dtype: int64
    
    === Transaction Data Quality ===
    Shape: (52930, 7)
    
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
    
    Rating stats:
    count    52930.000000
    mean         4.157699
    std          0.970543
    min          1.000000
    25%          4.000000
    50%          4.000000
    75%          5.000000
    max          5.000000
    Name: Rating, dtype: float64
    
    === Visit Data Quality ===
    Shape: (6, 2)
    
    Missing values:
    VisitModeId    0
    VisitMode      0
    dtype: int64
    
    Data types:
    int64     1
    object    1
    Name: count, dtype: int64
    
    === Attraction Data Quality ===
    Shape: (17, 2)
    
    Missing values:
    AttractionTypeId    0
    AttractionType      0
    dtype: int64
    
    Data types:
    int64     1
    object    1
    Name: count, dtype: int64
    

## **2. Data Cleaning and Standardization**

### **2.1 Handling Missing Values**


```python
# User data - drop rows with missing CityId (only 4 out of 33,530)
print("\nHandling missing values in user data...")
user_data.dropna(subset=['CityId'], inplace=True)

# City data - drop rows with missing CityName
print("\nHandling missing values in city data...")
city_data.dropna(subset=['CityName'], inplace=True)

# Transaction data - remove invalid ratings (1-5 scale)
print("\nCleaning rating values in transaction data...")
initial_count = len(transaction_data)
transaction_data = transaction_data[transaction_data['Rating'].between(1, 5)]
print(f"\nRemoved {initial_count - len(transaction_data)} invalid ratings")
```

    
    Handling missing values in user data...
    
    Handling missing values in city data...
    
    Cleaning rating values in transaction data...
    
    Removed 0 invalid ratings
    

### **2.2 Standardizing Categorical Variables**


```python
# Standardize VisitMode
print("\nStandardizing VisitMode...")
visit_data['VisitMode'] = visit_data['VisitMode'].str.strip().str.title()
print("VisitMode values after standardization:")
print(visit_data['VisitMode'].value_counts())

# Standardize AttractionType
print("\nStandardizing AttractionType...")
attraction_data['AttractionType'] = attraction_data['AttractionType'].str.strip().str.title()
print("AttractionType values after standardization:")
print(attraction_data['AttractionType'].value_counts())

# Standardize City Names
print("\nStandardizing City Names...")
city_name_mapping = {
    'New York City': 'New York', 'NYC': 'New York',
    'San Fran': 'San Francisco', 'S.F.': 'San Francisco',
    'Los Ang': 'Los Angeles', 'L.A.': 'Los Angeles',
    'Chicago, Il': 'Chicago', 'Chi-Town': 'Chicago',
    'Vegas': 'Las Vegas', 'D.C.': 'Washington DC',
    'Wash D.C.': 'Washington DC', 'Philly': 'Philadelphia',
    'Nola': 'New Orleans', 'Saint Louis': 'St. Louis',
    'St Louis': 'St. Louis', 'Ft Worth': 'Fort Worth',
    'Ft. Worth': 'Fort Worth', 'San Antone': 'San Antonio',
    'Atl': 'Atlanta', 'H-Town': 'Houston',
    'Big D': 'Dallas', 'Motor City': 'Detroit',
    'Beantown': 'Boston', '-': None
}

city_data['CityName'] = (
    city_data['CityName']
    .str.strip()
    .str.title()
    .replace(city_name_mapping)
    .str.replace(r'\bSt\b\.?', 'Saint', regex=True)
    .str.replace(r'\.', '', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)

# Drop any remaining null city names
city_data.dropna(subset=['CityName'], inplace=True)
```

    
    Standardizing VisitMode...
    VisitMode values after standardization:
    VisitMode
    -           1
    Business    1
    Couples     1
    Family      1
    Friends     1
    Solo        1
    Name: count, dtype: int64
    
    Standardizing AttractionType...
    AttractionType values after standardization:
    AttractionType
    Ancient Ruins                     1
    Ballets                           1
    Beaches                           1
    Caverns & Caves                   1
    Flea & Street Markets             1
    Historic Sites                    1
    History Museums                   1
    National Parks                    1
    Nature & Wildlife Areas           1
    Neighborhoods                     1
    Points Of Interest & Landmarks    1
    Religious Sites                   1
    Spas                              1
    Speciality Museums                1
    Volcanos                          1
    Water Parks                       1
    Waterfalls                        1
    Name: count, dtype: int64
    
    Standardizing City Names...
    

### **2.3 Standardizing Date Format**


```python
# Create proper date column in transaction data
print("\nStandardizing date format...")
transaction_data['VisitDate'] = pd.to_datetime(
    transaction_data['VisitYear'].astype(str) + '-' + 
    transaction_data['VisitMonth'].astype(str) + '-01'
)
print("Sample dates after standardization:")
print(transaction_data[['VisitYear', 'VisitMonth', 'VisitDate']].head())

```

    
    Standardizing date format...
    Sample dates after standardization:
       VisitYear  VisitMonth  VisitDate
    0       2022          10 2022-10-01
    1       2022          10 2022-10-01
    2       2022          10 2022-10-01
    3       2022          10 2022-10-01
    4       2022          10 2022-10-01
    

### **2.4 Verifying Referential Integrity**


```python
# Check for orphaned records
print("\nChecking referential integrity...")
print("User IDs in transactions but not in user data:", 
      set(transaction_data['UserId']) - set(user_data['UserId']))
print("City IDs in user but not in city data:", 
      set(user_data['CityId']) - set(city_data['CityId']))
print("Attraction IDs in transactions but not in item data:", 
      set(transaction_data['AttractionId']) - set(item_data['AttractionId']))

# Remove orphaned transactions (users not in user_data)
orphaned_users = {17595, 56972, 67461, 7175}
transaction_data = transaction_data[~transaction_data['UserId'].isin(orphaned_users)]
print(f"Removed {len(orphaned_users)} orphaned transactions")
```

    
    Checking referential integrity...
    User IDs in transactions but not in user data: {17595, 56972, 67461, 7175}
    City IDs in user but not in city data: set()
    Attraction IDs in transactions but not in item data: set()
    Removed 4 orphaned transactions
    

## **3. Feature Engineering**

### **3.1 Encoding Categorical Variables**


```python
# One-hot encode VisitMode
print("\nEncoding categorical variables...")
visit_encoded = pd.get_dummies(visit_data, columns=['VisitMode'], prefix='VisitMode')

# Frequency encoding for Country
country_counts = user_data['CountryId'].value_counts(normalize=True)
user_data['Country_freq_encoded'] = user_data['CountryId'].map(country_counts)

# One-hot encode AttractionType
attraction_encoded = pd.get_dummies(attraction_data, columns=['AttractionType'], prefix='Attraction')
```

    
    Encoding categorical variables...
    

### **3.2 Data Consolidation**


```python
print("\nMerging datasets...")
# Merge transaction with user data
merged_df = pd.merge(transaction_data, user_data, on='UserId')

# Merge with city data
merged_df = pd.merge(merged_df, city_data, left_on='CityId', right_on='CityId', suffixes=('_user', '_city'))

# Merge with item and attraction data
merged_df = pd.merge(merged_df, item_data, on='AttractionId')
merged_df = pd.merge(merged_df, attraction_data, on='AttractionTypeId')

# Merge with visit mode data
merged_df = pd.merge(merged_df, visit_data, left_on='VisitMode', right_on='VisitModeId')


```

    
    Merging datasets...
    

### **3.3 Creating User-Level Features**


```python
print("\nCreating user-level features...")
# Calculate user-level statistics
user_features = merged_df.groupby('UserId').agg({
    'Rating': ['mean', 'count', 'std'],
    'VisitMode_y': lambda x: x.mode()[0],
    'AttractionId': 'nunique'
}).reset_index()

# Flatten multi-index columns
user_features.columns = ['UserId', 'AvgRating', 'TotalVisits', 'RatingStd', 'FrequentVisitMode', 'UniqueAttractions']

# Calculate visit mode proportions
visit_mode_counts = merged_df.groupby(['UserId', 'VisitMode_y']).size().unstack(fill_value=0)
visit_mode_proportions = visit_mode_counts.div(visit_mode_counts.sum(axis=1), axis=0)
visit_mode_proportions = visit_mode_proportions.add_prefix('VisitModeProp_')

# Combine with main dataframe
merged_df = pd.merge(merged_df, user_features, on='UserId')
merged_df = pd.merge(merged_df, visit_mode_proportions, left_on='UserId', right_index=True)

```

    
    Creating user-level features...
    

### **3.4 Final Feature Engineering**


```python
print("\nFinal feature engineering...")
# One-hot encoding
merged_df = pd.get_dummies(merged_df, columns=[
    'VisitMode_y', 'AttractionType', 'ContinentId'
], drop_first=True)

# Temporal features
merged_df['VisitYear'] = merged_df['VisitDate'].dt.year
merged_df['VisitMonth'] = merged_df['VisitDate'].dt.month
merged_df['VisitDayOfWeek'] = merged_df['VisitDate'].dt.dayofweek
merged_df['IsWeekend'] = merged_df['VisitDayOfWeek'].isin([5, 6]).astype(int)

# Normalization
from sklearn.preprocessing import MinMaxScaler

numerical_features = ['Rating', 'AvgRating', 'TotalVisits', 'UniqueAttractions']
scaler = MinMaxScaler()
merged_df[numerical_features] = scaler.fit_transform(merged_df[numerical_features])

```

    
    Final feature engineering...
    

## **4. Final Dataset Preparation**


```python
# Select features for modeling
print("\nSelecting final features...")
selected_features = [
    # User features
    'AvgRating', 'TotalVisits', 'RatingStd', 'UniqueAttractions',
    
    # Visit behavior
    *[col for col in merged_df.columns if col.startswith('VisitModeProp_')],
    *[col for col in merged_df.columns if col.startswith('VisitMode_y_')],
    
    # Attraction features
    *[col for col in merged_df.columns if col.startswith('AttractionType_')],
    
    # Location features
    *[col for col in merged_df.columns if col.startswith('ContinentId_')],
    'Country_freq_encoded',
    
    # Temporal features
    'VisitYear', 'VisitMonth', 'IsWeekend'
]

modeling_df = merged_df[selected_features + ['Rating']]

# Handle any remaining missing values
print("\nHandling missing values in final dataset...")
numerical_cols = ['AvgRating', 'TotalVisits', 'RatingStd']
modeling_df[numerical_cols] = modeling_df[numerical_cols].fillna(
    modeling_df[numerical_cols].median()
)

visit_mode_cols = [col for col in modeling_df.columns if 'VisitModeProp_' in col]
modeling_df[visit_mode_cols] = modeling_df[visit_mode_cols].fillna(0)

modeling_df['Rating'] = modeling_df['Rating'].fillna(modeling_df['Rating'].median())

```

    
    Selecting final features...
    
    Handling missing values in final dataset...
    

    C:\Users\Admin\AppData\Local\Temp\ipykernel_2724\781340121.py:27: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      modeling_df[numerical_cols] = modeling_df[numerical_cols].fillna(
    C:\Users\Admin\AppData\Local\Temp\ipykernel_2724\781340121.py:32: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      modeling_df[visit_mode_cols] = modeling_df[visit_mode_cols].fillna(0)
    C:\Users\Admin\AppData\Local\Temp\ipykernel_2724\781340121.py:34: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      modeling_df['Rating'] = modeling_df['Rating'].fillna(modeling_df['Rating'].median())
    

### **Final Verification**


```python
print("\n=== Final Dataset Summary ===")
print("Shape:", modeling_df.shape)
print("\nFirst 5 columns and last 5 columns:")
print(modeling_df.columns[:5].tolist(), "...", modeling_df.columns[-5:].tolist())
print("\nData types:")
print(modeling_df.dtypes.value_counts())
print("\nMissing values:", modeling_df.isna().sum().sum())
print("\nNormalized features range check:")
print(modeling_df[numerical_features].describe().loc[['min', 'max']])

# Save final dataset
print("\nSaving final dataset...")
modeling_df.to_csv('preprocessed_tourism_data.csv', index=False)
print("Preprocessing complete! Final dataset saved as 'preprocessed_tourism_data.csv'")

```

    
    === Final Dataset Summary ===
    Shape: (52922, 38)
    
    First 5 columns and last 5 columns:
    ['AvgRating', 'TotalVisits', 'RatingStd', 'UniqueAttractions', 'VisitModeProp_Business'] ... ['Country_freq_encoded', 'VisitYear', 'VisitMonth', 'IsWeekend', 'Rating']
    
    Data types:
    bool       24
    float64    11
    int32       2
    int64       1
    Name: count, dtype: int64
    
    Missing values: 0
    
    Normalized features range check:
         Rating  AvgRating  TotalVisits  UniqueAttractions
    min     0.0        0.0          0.0                0.0
    max     1.0        1.0          1.0                1.0
    
    Saving final dataset...
    Preprocessing complete! Final dataset saved as 'preprocessed_tourism_data.csv'
    
