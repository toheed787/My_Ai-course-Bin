#Area of rectangle
Length = float(input("Enter length")) 
Width = float(input("Enter width"))
area = Length * Width
print("Area of rectangle is", area)

# A power B
a = int(input("Enter First Number"))
b = int(input("Enter Second Number"))
print("Result", a**b)

#the difference between / (division) and // (floor division)
print("Division", 10 / 3)
print("Float Division", 10 // 3)

# Use the modulus operator %
print("Remainder" , 25 % 4)

#The average of five numbers
a = float(input("Enter number 1 "))
b = float(input("Enter number 2 "))
c = float(input("Enter number 3 "))
d = float(input("Enter number 4 "))
e = float(input("Enter number 5 "))
avg = (a + b + c + d + e) / 5
print("Average", avg)

# program that converts minutes into hours
minutes = int(input("Enter minutes "))
hours = minutes // 60
remaining = minutes % 60
print("Hours", hours, "Minutes", remaining)

#Area of circle
r = float(input("Enter radius "))
area = 3.14 * r * r
print("Area of circle", area)

#Cube of a number
num = int(input("Enter number "))
print("Cube", num ** 3)

# EMDAS demonstration
result = 10 + 5 * 2
print("Result", result)

# Simple Interest
P = float(input("Enter principal "))
R = float(input("Enter rate "))
T = float(input("Enter time "))
SI = (P * R * T) / 100
print("Simple Interest:", SI)
# Part 3 ends here