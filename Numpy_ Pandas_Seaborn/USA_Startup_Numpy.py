import numpy as np

ids, price , long , lat = np.genfromtxt(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Numpy_ Pandas_Seaborn\USA_startup_csv.csv', delimiter=',', usecols=(2,3,4,8), unpack=True, dtype=None,skip_header=1)

print(ids)
print(price)
print(long)
print(lat)

# US start up growth investment statistics operations
print("Investment Mean:", np.mean(price))
print("Investment Average:", np.average(price))
print("Investment Std:", np.std(price))
print("Investment Median:", np.median(price))
print("Investment Percentile 25:", np.percentile(price,25))
print("Investment Percentile 75:", np.percentile(price,75))
print("Investment Percentile 3:", np.percentile(price,3))
print("Investment Min:", np.min(price))
print("Investment Max:", np.max(price))

# US start up growth  - maths operations
print("Investment Square:", np.square(price))
print("Investment Sqrt:", np.sqrt(price))
print("Investment Power:", np.power(price,2))
print("Investment Abs:", np.abs(price))



# Perform basic arithmetic operations
addition = long + lat
subtraction = long - lat
multiplication = long * lat
division = long / lat

print("Valuation + Growth:", addition)
print("Valuation - Growth:", subtraction)
print("Valuation * Growth:", multiplication)
print("Valuation / Growth:", division)


#Trigonometric Functions


pricePie = (price/np.pi) + 1

sine_values = np.sin(pricePie)
cosine_values = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("Sine:", sine_values)
print("Cosine:", cosine_values)
print("Tangent:", tangent_values)

print("Exponential:", np.exp(pricePie))


# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(pricePie)
log10_array = np.log10(pricePie)

print("Natural Log:", log_array)
print("Log10:", log10_array)

#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
print("Sinh:", np.sinh(pricePie))
print("Cosh:", np.cosh(pricePie))
print("Tanh:", np.tanh(pricePie))
print("Arcsinh:", np.arcsinh(pricePie))


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(pricePie)
# Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element

cosh_values = np.cosh(pricePie)

print("Startup Investment Amount - div - pi - Hyperbolic Cosine values:", cosh_values)

#Example: Hyperbolic Tangent
# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(pricePie)
print("Startup Investment Amount - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(pricePie)
print("Startup Investment Amount - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(pricePie)
print("Startup Investment Amount- div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)


# Startup Valuation + Growth Rate - 2D array
D2Data = np.array([long, lat])

print("Startup Valuation + Growth Rate - 2D array:", D2Data)
D2LongLat = np.array([long, lat])
print ("Startup Investment Amount Long Plus Lat - 2 dimentional arrary - " ,D2LongLat)

# check the dimension of array1
print("Startup Investment Amount Long Plus Lat - 2 dimentional arrary - dimension" , D2LongLat.ndim) 
# Output: 2

# return total number of elements in array1
print("Startup Investment Amount Long Plus Lat - 2 dimentional arrary - total number of elements" ,D2LongLat.size)
# Output: 6

# return a tuple that gives size of array in each dimension
print("Startup Investment Amount Long Plus Lat - 2 dimentional arrary - gives size of array in each dimension" ,D2LongLat.shape)
# Output: (2,3)

# check the data type of array1
print("Startup Investment Amount Long Plus Lat - 2 dimentional arrary - data type" ,D2LongLat.dtype) 
# Output: int64

# Splicing array
D2LongLatSlice=  D2LongLat[:1,:5]
print("Startup Investment Amount Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1,:5] " , D2LongLatSlice)
D2LongLatSlice2=  D2LongLat[:1, 4:15:4]
print("Startup Investment Amount Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1, 4:15:4] " , D2LongLatSlice2)



# Indexing array
D2LongLatSliceItemOnly=  D2LongLatSlice[0,1]
print("Startup Investment Amount Long Plus Lat - 2 dimentional arrary - Index array - D2LongLatSlice[1,5] " , D2LongLatSliceItemOnly)
D2LongLatSlice2ItemOnly=  D2LongLatSlice2[0, 2]
print("Startup Investment Amount Long Plus Lat - 2 dimentional arrary - index array - D2LongLatSlice2[0, 2] " , D2LongLatSlice2ItemOnly)


#You should use the builtin function nditer, if you don't need to have the indexes values.
for elem in np.nditer(D2LongLat):
    print(elem)

#EDIT: If you need indexes (as a tuple for 2D table), then:
for index, elem in np.ndenumerate(D2LongLat):
    print(index, elem)

"""# for loop
rows = np.shape(D2LongLat[0])[0]
cols = np.shape(D2LongLat[1])[0]
for i in range(0, (rows + 1)):
    for j in range(0, (cols + 1)):
        print (D2LongLat[i,j])
"""


# 2 x N  ========>>>>> 1 x (total elements) reshape

D2Data1D = np.reshape(D2Data, (1, D2Data.size))

print("Startup Valuation + Growth - 2D array reshape :", D2Data1D)

print("Startup Valuation + Growth - reshape Size :", D2Data1D.size)

print("Startup Valuation + Growth - reshape ndim :", D2Data1D.ndim)

print("Startup Valuation + Growth - reshape shape :", D2Data1D.shape)

print("Startup Valuation + Growth - reshape ndim :", D2Data1D.ndim)


print()