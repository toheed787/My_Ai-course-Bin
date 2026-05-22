import pandas as pd

# Read csv file to DataFrame
df = pd.read_csv(
    r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Numpy_ Pandas_Seaborn\USA_RealEstate_csv.csv',
    delimiter=",",
    parse_dates=['prev_sold_date']
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
agency = df['brokered_by']
print("access the brokered_by column:")
print(agency)
print()

# access multiple columns
agency_agent = df[['brokered_by', 'city']]
print("access multiple columns:")
print(agency_agent)
print()

# ---------------------------------------------------
# Case 1 : using .loc
# ---------------------------------------------------

# Selecting a single row using .loc
second_row = df.loc[1]
print("# Selecting a single row using .loc")
print(second_row)
print()

# Selecting multiple rows using .loc
second_row2 = df.loc[[1, 3]]
print("# Selecting multiple rows using .loc")
print(second_row2)
print()

# Selecting a slice of rows using .loc
second_row3 = df.loc[1:5]
print("# Selecting a slice of rows using .loc")
print(second_row3)
print()

# Conditional selection of rows using .loc
second_row4 = df.loc[df['brokered_by'] == df['brokered_by'].iloc[0]]
print("# Conditional selection of rows using .loc")
print(second_row4)
print()

# Selecting a single column using .loc
second_row5 = df.loc[:1, 'brokered_by']
print("# Selecting a single column using .loc")
print(second_row5)
print()

# Selecting multiple columns using .loc
second_row6 = df.loc[:1, ['brokered_by', 'city']]
print("# Selecting multiple columns using .loc")
print(second_row6)
print()

# Selecting a slice of columns using .loc
second_row7 = df.loc[:1, 'city':'zip_code']
print("# Selecting a slice of columns using .loc")
print(second_row7)
print()

# Combined row and column selection using .loc
second_row8 = df.loc[df['brokered_by'] == df['brokered_by'].iloc[0], 'city':'zip_code']
print("# Combined row and column selection using .loc")
print(second_row8)
print()

# ---------------------------------------------------
# Case 2 : using .loc with index_col
# ---------------------------------------------------

print("# Case 2 : using .loc with index_col - starts here")

df_index_col = pd.read_csv(
   r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Numpy_ Pandas_Seaborn\USA_RealEstate_csv.csv',
    delimiter=",",
    parse_dates=['prev_sold_date'],
    index_col='street'
)
df_index_col = df_index_col.sort_index()
print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())

# Selecting a single row using .loc
first_zip = df_index_col.index[0]
second_row = df_index_col.loc[first_zip]
print("# Selecting a single row using .loc")
print(second_row)
print()

# Selecting multiple rows using .loc
second_row2 = df_index_col.loc[df_index_col.index[:2]]
print("# Selecting multiple rows using .loc")
print(second_row2)
print()

# Selecting a slice of rows using .loc
second_row3 = df_index_col[
    (df_index_col.index >= df_index_col.index.min()) &
    (df_index_col.index <= df_index_col.index.max())
]
print("# Selecting a slice of rows using .loc")
print(second_row3.head())
print()

# Conditional selection of rows using .loc
second_row4 = df_index_col.loc[df_index_col['brokered_by'] == df_index_col['brokered_by'].iloc[0]]
print("# Conditional selection of rows using .loc")
print(second_row4)
print()

# Selecting a single column using .loc
second_row5 = df_index_col.loc[:, 'brokered_by']
print("# Selecting a single column using .loc")
print(second_row5.head())
print()

# Selecting multiple columns using .loc
second_row6 = df_index_col.loc[:, ['brokered_by', 'city']]
print("# Selecting multiple columns using .loc")
print(second_row6.head())
print()

# Selecting a slice of columns using .loc
second_row7 = df_index_col.loc[:, 'city':'zip_code']
print("# Selecting a slice of columns using .loc")
print(second_row7.head())
print()

# Combined row and column selection using .loc
second_row8 = df_index_col.loc[
    df_index_col['brokered_by'] == df_index_col['brokered_by'].iloc[0],
    'city':'zip_code'
]
print("# Combined row and column selection using .loc")
print(second_row8)
print()

# ---------------------------------------------------
# Case 3 : Using .iloc
# ---------------------------------------------------

print("# Case 3 : Using .iloc - starts here")

# Selecting a single row using .iloc
second_row = df_index_col.iloc[0]
print("# Selecting a single row using .iloc")
print(second_row)
print()

# Selecting multiple rows using .iloc
second_row2 = df_index_col.iloc[[1, 3, 5]]
print("# Selecting multiple rows using .iloc")
print(second_row2)
print()

# Selecting a slice of rows using .iloc
second_row3 = df_index_col.iloc[2:5]
print("# Selecting a slice of rows using .iloc")
print(second_row3)
print()

# Selecting a single column using .iloc
second_row5 = df_index_col.iloc[:, 2]
print("# Selecting a single column using .iloc")
print(second_row5)
print()

# Selecting multiple columns using .iloc
second_row6 = df_index_col.iloc[:, [2, 4]]
print("# Selecting multiple columns using .iloc")
print(second_row6)
print()

# Selecting a slice of columns using .iloc
second_row7 = df_index_col.iloc[:, 2:4]
print("# Selecting a slice of columns using .iloc")
print(second_row7)
print()

# Combined row and column selection using .iloc
second_row8 = df_index_col.iloc[[1, 3, 5], 2:4]
print("# Combined row and column selection using .iloc")
print(second_row8)
print()

# ---------------------------------------------------
# DataFrame Manipulation
# ---------------------------------------------------

print("Next Run")

# Add a new row
df.loc[len(df.index)] = [
    "Test Broker", "for_sale", 500000, 3, 2, 0.5,
    "Test Street", "Test City", "Test State", 12345,
    1500, "2020-01-01"
]

print("Modified DataFrame - add a new row:")
print(df)
print()

# Remove rows
df.drop(1, axis=0, inplace=True)
df.drop(index=2, inplace=True)
df.drop([3, 5], axis=0, inplace=True)

print("Modified DataFrame - Remove Rows:")
print(df)

# Remove columns
df.drop('street', axis=1, inplace=True)
df.drop(columns='status', inplace=True)
df.drop(['city'], axis=1, inplace=True)

print("Modified DataFrame - delete columns:")
print(df)

# Rename labels
df.rename(columns={'state': 'state_changed'}, inplace=True)
df.rename(
    mapper={'bed': 'bed_changed', 'prev_sold_date': 'prev_sold_date_changed'},
    axis=1,
    inplace=True
)

print("Modified DataFrame - Rename Labels:")
print(df)

# Rename row labels
df.rename(index={0: 7}, inplace=True)
df.rename(mapper={1: 10, 2: 100}, axis=0, inplace=True)

print("Modified DataFrame - Rename Row Labels:")
print(df)

# ---------------------------------------------------
# query()
# ---------------------------------------------------

selected_rows = df.query('price > 11000000')
print(selected_rows.to_string())
print(len(selected_rows))

# ---------------------------------------------------
# Sorting
# ---------------------------------------------------

sorted_df = df.sort_values(by='price')
print(sorted_df.to_string(index=False))

df1 = df.sort_values(by=['price', 'zip_code'])

print("Sorting by 'price' and then by 'zip_code':")
print(df1.to_string(index=False))

# ---------------------------------------------------
# groupby
# ---------------------------------------------------

grouped = df.groupby('zip_code')['price'].sum()

print(grouped.to_string())
print("grouped:", len(grouped))

# ---------------------------------------------------
# Data Cleaning
# ---------------------------------------------------

df_cleaned = df.dropna()
print("Cleaned Data:\n", df_cleaned)

# filling NaN values
df.fillna(0, inplace=True)

print("\nData after filling NaN with 0:\n", df)

# ---------------------------------------------------
# pandas.array
# ---------------------------------------------------

data = [2, 4, 6, 8]
array1 = pd.array(data)
print(array1)

int_array = pd.array([1, 2, 3, 4, 5], dtype='int')
print(int_array)
print()