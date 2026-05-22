import numpy as np

ids, price , long , lat = np.genfromtxt(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Numpy_ Pandas_Seaborn\USA_RealEstate_csv.csv', delimiter=',', usecols=(0,2,5,10), unpack=True, dtype=None,skip_header=1)

print(ids)
print(price)
print(long)
print(lat)

# RealEstate price  - statistics operations
print("RealEstate Price mean: " , np.mean(price))
print("RealEstate Price average: " , np.average(price))
print("RealEstate Price std: " , np.std(price))
print("RealEstate Price mod: " , np.median(price))
print("RealEstate Price percentile - 25: " , np.percentile(price,25))
print("RealEstate Price percentile  - 75: " , np.percentile(price,75))
print("RealEstatePrice percentile  - 3: " , np.percentile(price,3))
print("RealEstate Price min : " , np.min(price))
print("RealEstate Price max : " , np.max(price))

# Zameen.com price  - maths operations
print("RealEstate Price square: " , np.square(price))
print("RealEstate Price sqrt: " , np.sqrt(price))
print("RealEstate Price pow: " , np.power(price,price))
print("RealEstate Price abs: " , np.abs(price))



# Perform basic arithmetic operations
addition = long + lat
subtraction = long - lat
multiplication = long * lat
division = long / lat

print("RealEstate  Long - lat - Addition:", addition)
print(" RealEstate Long - lat - Subtraction:", subtraction)
print(" RealEstate Long - lat - Multiplication:", multiplication)
print(" RealEstate Long - lat - Division:", division)


#Trigonometric Functions

pricePie = (price/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(pricePie)
cosine_values = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("RealEstate Price - div - pie  - Sine values:", sine_values)
print("RealEstatePrice - div - pie Cosine values:", cosine_values)
print("RealEstate Price - div - pie Tangent values:", tangent_values)

print("v Price - div - pie  - Exponential values:", np.exp(pricePie))


# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(pricePie)
log10_array = np.log10(pricePie)

print("RealEstate Price - div - pie  - Natural logarithm values:", log_array)
print("RealEstate Price - div - pie  = Base-10 logarithm values:", log10_array)

#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(pricePie)
print("RealEstate Price - div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(pricePie)
print("RealEstate Price - div - pie   - Hyperbolic Cosine values:", cosh_values)

#Example: Hyperbolic Tangent
# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(pricePie)
print("RealEstate Price - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(pricePie)
print("RealEstate Price - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(pricePie)
print("RealEstate Price - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)


#RealEstae Long Plus Lat - 2 dimentional arrary
D2LongLat = np.array([long,
                  lat])

print ("RealEstate Long Plus Lat - 2 dimentional arrary - " ,D2LongLat)

# check the dimension of array1
print("RealEstate Long Plus Lat - 2 dimentional arrary - dimension" , D2LongLat.ndim) 
# Output: 2

# return total number of elements in array1
print("RealEstate Long Plus Lat - 2 dimentional arrary - total number of elements" ,D2LongLat.size)
# Output: 6

# return a tuple that gives size of array in each dimension
print("RealEstate Long Plus Lat - 2 dimentional arrary - gives size of array in each dimension" ,D2LongLat.shape)
# Output: (2,3)

# check the data type of array1
print("RealEstate Long Plus Lat - 2 dimentional arrary - data type" ,D2LongLat.dtype) 
# Output: int64

# Splicing array
D2LongLatSlice=  D2LongLat[:1,:5]
print("RealEstate Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1,:5] " , D2LongLatSlice)
D2LongLatSlice2=  D2LongLat[:1, 4:15:4]
print("RealEstate Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1, 4:15:4] " , D2LongLatSlice2)



# Indexing array
D2LongLatSliceItemOnly=  D2LongLatSlice[0,1]
print("RealEstateLong Plus Lat - 2 dimentional arrary - Index array - D2LongLatSlice[1,5] " , D2LongLatSliceItemOnly)
D2LongLatSlice2ItemOnly=  D2LongLatSlice2[0, 2]
print("RealEstate Long Plus Lat - 2 dimentional arrary - index array - D2LongLatSlice2[0, 2] " , D2LongLatSlice2ItemOnly)


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


# reshape automatically according to your dataset size
D2LongLat1TO = np.reshape(D2LongLat, (1, D2LongLat.size))

print("reshape : " , D2LongLat1TO)
print("Size " , D2LongLat1TO.size)
print("ndim " , D2LongLat1TO.ndim)
print("shape " , D2LongLat1TO.shape)
print("ndim " , D2LongLat1TO.ndim)



print()