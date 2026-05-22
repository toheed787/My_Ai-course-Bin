import pandas as pd

# Read csv file to DataFrame
df = pd.read_csv(
    r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Numpy_ Pandas_Seaborn\USA_RealEstate_Second_csv.csv',
    delimiter=",",
    parse_dates=['Date Recorded']
)

print(df)

print("df - data types", df.dtypes)

print("df.info():", df.info())

# display the last three rows
print('Last three Rows:')
print(df.tail(3))

# display the first three rows
print('First Three Rows:')
print(df.head(3))
print()

# Summary statistics
print("Summary of Statistics of DataFrame using describe() method")
print(df.describe())

# shape
print("Counting the rows and columns in DataFrame using shape():", df.shape)
print()

# access single column
agency = df['Town']
print("access the Town column:")
print(agency)
print()

# access multiple columns
agency_agent = df[['Town', 'Property Type']]
print("access multiple columns:")
print(agency_agent)
print()

# ---------------------------------------------------
# Case 1 : using .loc
# ---------------------------------------------------

second_row = df.loc[1]
print("# Selecting a single row using .loc")
print(second_row)
print()

second_row2 = df.loc[[1, 3]]
print("# Selecting multiple rows using .loc")
print(second_row2)
print()

second_row3 = df.loc[1:5]
print("# Selecting a slice of rows using .loc")
print(second_row3)
print()

second_row4 = df.loc[df['Town'] == df['Town'].iloc[0]]
print("# Conditional selection of rows using .loc")
print(second_row4)
print()

second_row5 = df.loc[:1, 'Town']
print("# Selecting a single column using .loc")
print(second_row5)
print()

second_row6 = df.loc[:1, ['Town', 'Property Type']]
print("# Selecting multiple columns using .loc")
print(second_row6)
print()

second_row7 = df.loc[:1, 'Town':'Property Type']
print("# Selecting a slice of columns using .loc")
print(second_row7)
print()

second_row8 = df.loc[df['Town'] == df['Town'].iloc[0], 'Town':'Property Type']
print("# Combined row and column selection using .loc")
print(second_row8)
print()

# ---------------------------------------------------
# Case 2 : using .loc with index_col
# ---------------------------------------------------

print("# Case 2 : using .loc with index_col - starts here")

df_index_col = pd.read_csv(
    r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Numpy_ Pandas_Seaborn\USA_RealEstate_Second_csv.csv',
    delimiter=",",
    parse_dates=['Date Recorded'],
    index_col='Address'
)

df_index_col = df_index_col.sort_index()
print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())

first_zip = df_index_col.index[0]
second_row = df_index_col.loc[first_zip]
print("# Selecting a single row using .loc")
print(second_row)
print()

second_row2 = df_index_col.loc[df_index_col.index[:2]]
print("# Selecting multiple rows using .loc")
print(second_row2)
print()

second_row3 = df_index_col[
    (df_index_col.index >= df_index_col.index.min()) &
    (df_index_col.index <= df_index_col.index.max())
]
print("# Selecting a slice of rows using .loc")
print(second_row3.head())
print()

second_row4 = df_index_col.loc[df_index_col['Town'] == df_index_col['Town'].iloc[0]]
print("# Conditional selection of rows using .loc")
print(second_row4)
print()

second_row5 = df_index_col.loc[:, 'Town']
print("# Selecting a single column using .loc")
print(second_row5.head())
print()

second_row6 = df_index_col.loc[:, ['Town', 'Property Type']]
print("# Selecting multiple columns using .loc")
print(second_row6.head())
print()

second_row7 = df_index_col.loc[:, 'Town':'Property Type']
print("# Selecting a slice of columns using .loc")
print(second_row7.head())
print()

second_row8 = df_index_col.loc[
    df_index_col['Town'] == df_index_col['Town'].iloc[0],
    'Town':'Property Type'
]
print("# Combined row and column selection using .loc")
print(second_row8)
print()

# Case 3 : Using .iloc

print("# Case 3 : Using .iloc - starts here")

second_row = df_index_col.iloc[0]
print("# Selecting a single row using .iloc")
print(second_row)
print()

second_row2 = df_index_col.iloc[[1, 3, 5]]
print("# Selecting multiple rows using .iloc")
print(second_row2)
print()

second_row3 = df_index_col.iloc[2:5]
print("# Selecting a slice of rows using .iloc")
print(second_row3)
print()

second_row5 = df_index_col.iloc[:, 2]
print("# Selecting a single column using .iloc")
print(second_row5)
print()

second_row6 = df_index_col.iloc[:, [2, 4]]
print("# Selecting multiple columns using .iloc")
print(second_row6)
print()

second_row7 = df_index_col.iloc[:, 2:4]
print("# Selecting a slice of columns using .iloc")
print(second_row7)
print()

second_row8 = df_index_col.iloc[[1, 3, 5], 2:4]
print("# Combined row and column selection using .iloc")
print(second_row8)
print()

# DataFrame Manipulation

print("Next Run")

df.loc[len(df.index)] = {
    "Serial Number": 999999,
    "List Year": 2022,
    "Date Recorded": pd.to_datetime("2022-01-01"),
    "Town": "Test Town",
    "Address": "Test Address",
    "Assessed Value": 100000,
    "Sale Amount": 120000,
    "Sales Ratio": 1.2,
    "Property Type": "Residential",
    "Residential Type": "Single Family"
}

print("Modified DataFrame - add a new row:")
print(df)
print()

df.drop(1, axis=0, inplace=True)
df.drop(index=2, inplace=True)

print("Modified DataFrame - Remove Rows:")
print(df)

df.drop('Address', axis=1, inplace=True)

print("Modified DataFrame - delete columns:")
print(df)

df.rename(columns={'Town': 'Town_Changed'}, inplace=True)

print("Modified DataFrame - Rename Labels:")
print(df)

# query()

selected_rows = df.query('`Sale Amount` > 1000000')
print(selected_rows.to_string())
print(len(selected_rows))

# Sorting

sorted_df = df.sort_values(by='Sale Amount')
print(sorted_df.to_string(index=False))

df1 = df.sort_values(by=['Sale Amount', 'Assessed Value'])

print("Sorting by 'Sale Amount' and then by 'Assessed Value':")
print(df1.to_string(index=False))

# ---------------------------------------------------
# groupby
# ---------------------------------------------------

grouped = df.groupby('Town_Changed')['Sale Amount'].sum()
print(grouped.to_string())
print("grouped:", len(grouped))

# ---------------------------------------------------
# Data Cleaning
# ---------------------------------------------------

df_cleaned = df.dropna()
print("Cleaned Data:\n", df_cleaned)

# df.fillna(0, inplace=True)   

numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

text_cols = df.select_dtypes(include=['object', 'string']).columns
df[text_cols] = df[text_cols].fillna("Unknown")

print("\nData after filling NaN with 0:\n", df)

# ---------------------------------------------------
# pandas.array
# ---------------------------------------------------

data = [2, 4, 6, 8]
array1 = pd.array(data)
print(array1)

int_array = pd.array([1, 2, 3, 4, 5], dtype='int')
print(int_array)