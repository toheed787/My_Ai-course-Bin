import pandas as pd

# ---------------------------------------------------
# Read CSV File
# ---------------------------------------------------

df = pd.read_csv(
    r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Numpy_ Pandas_Seaborn\USA_startup_csv.csv',
    delimiter=","
)
print(df.columns)
print(df)
print("df - data types\n", df.dtypes)
print("df.info():")
print(df.info())

# ---------------------------------------------------
# Basic Exploration
# ---------------------------------------------------

print('Last three Rows:')
print(df.tail(3))

print('First Three Rows:')
print(df.head(3))
print()

print("Summary of Statistics using describe()")
print(df.describe())

print("Shape of DataFrame:", df.shape)
print()

# ---------------------------------------------------
# Column Access
# ---------------------------------------------------

industry = df['Industry']
print("Access the Industry column:")
print(industry)
print()

industry_funding = df[['Industry', 'Funding Rounds']]
print("Access multiple columns:")
print(industry_funding)
print()

# ---------------------------------------------------
# Case 1 : using .loc
# ---------------------------------------------------

print("# Case 1: using .loc")

print(df.loc[1])
print(df.loc[[1, 3]])
print(df.loc[1:5])

print(df.loc[df['Industry'] == df['Industry'].iloc[0]])

print(df.loc[:1, 'Industry'])
print(df.loc[:1, ['Industry', 'Funding Rounds']])
print(df.loc[:1, 'Industry':'Funding Rounds'])

print(df.loc[
    df['Industry'] == df['Industry'].iloc[0],
    'Industry':'Funding Rounds'
])
print()

# ---------------------------------------------------
# Case 2 : using .loc with index_col
# ---------------------------------------------------

print("# Case 2: using .loc with index_col")

df_index_col = pd.read_csv(
    r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Numpy_ Pandas_Seaborn\USA_startup_csv.csv',
    delimiter=",",
    index_col='Startup Name'
)

df_index_col = df_index_col.sort_index()

print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())

first_index = df_index_col.index[0]
print(df_index_col.loc[first_index])
print(df_index_col.loc[df_index_col.index[:2]])

print(df_index_col[
    (df_index_col.index >= df_index_col.index.min()) &
    (df_index_col.index <= df_index_col.index.max())
].head())

print(df_index_col.loc[
    df_index_col['Industry'] == df_index_col['Industry'].iloc[0]
])

print(df_index_col.loc[:, 'Industry'])
print(df_index_col.loc[:, ['Industry', 'Funding Rounds']])
print(df_index_col.loc[:, 'Industry':'Funding Rounds'])

print(df_index_col.loc[
    df_index_col['Industry'] == df_index_col['Industry'].iloc[0],
    'Industry':'Funding Rounds'
])
print()

# ---------------------------------------------------
# Case 3 : Using .iloc
# ---------------------------------------------------

print("# Case 3: using .iloc")

print(df_index_col.iloc[0])
print(df_index_col.iloc[[1, 3, 5]])
print(df_index_col.iloc[2:5])
print(df_index_col.iloc[:, 2])
print(df_index_col.iloc[:, [2, 4]])
print(df_index_col.iloc[:, 2:4])
print(df_index_col.iloc[[1, 3, 5], 2:4])
print()

# ---------------------------------------------------
# DataFrame Manipulation
# ---------------------------------------------------

print("Next Run")

df.loc[len(df.index)] = {
    "Startup Name": "Test Startup",
    "Industry": "AI",
    "Funding Amount": 5000000,
    "Valuation": 20000000,
    "Investment Round": "Series A",
    "Country": "USA",
    "Date": "2024-01-01"
}

print("After Adding Row:")
print(df)
print()

df.drop([1, 2], axis=0, inplace=True)

print("After Removing Rows:")
print(df)
print()

df.drop('Country', axis=1, inplace=True)

print("After Removing Column:")
print(df)
print()

df.rename(columns={'Industry': 'Industry_Changed'}, inplace=True)

print("After Renaming Column:")
print(df)
print()

# ---------------------------------------------------
# query()
# ---------------------------------------------------

selected_rows = df.query('`Funding Rounds` > 1000000')
print(selected_rows.to_string())
print("Selected rows:", len(selected_rows))
print()

# ---------------------------------------------------
# Sorting
# ---------------------------------------------------

sorted_df = df.sort_values(by='Funding Rounds')
print(sorted_df.to_string(index=False))

df1 = df.sort_values(by=['Funding Rounds', 'Valuation (USD)'])
print("Sorting by Funding Rounds and Valuation (USD):")
print(df1.to_string(index=False))
print()

# ---------------------------------------------------
# groupby
# ---------------------------------------------------

grouped = df.groupby('Industry_Changed')['Funding Rounds'].sum()
print(grouped.to_string())
print("Group count:", len(grouped))
print()

# ---------------------------------------------------
# Data Cleaning
# ---------------------------------------------------

df_cleaned = df.dropna()
print("Cleaned Data:\n", df_cleaned)

# Safe fillna
df[df.select_dtypes(include='number').columns] = \
    df.select_dtypes(include='number').fillna(0)

df[df.select_dtypes(include='object').columns] = \
    df.select_dtypes(include='object').fillna("Unknown")

print("\nAfter Filling NaN:\n", df)

# ---------------------------------------------------
# pandas.array
# ---------------------------------------------------

data = [2, 4, 6, 8]
array1 = pd.array(data)
print(array1)

int_array = pd.array([1, 2, 3, 4, 5], dtype='int')
print(int_array)