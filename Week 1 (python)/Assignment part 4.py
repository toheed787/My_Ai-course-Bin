#Compare two numbers entered by the user
a = int(input("Enter first number"))
b = int(input("Enter second number"))
print("First is greater than second", a > b )

# Check if a user-entered number is even
a = int(input("Enter a number"))
print("Is even", a % 2 == 0)

# A number is between 10 and 50
num = int(input("Enter a number"))
print("Between 10 and 50", num >= 10 and num <= 50)

# Check if a string entered
a = input("Enter a word")
print(a == "PYTHON")

#  Use the or operator
user = input("Enter username ")
print(user == "Admin" or user == "Superuser")

# Demonstrate the not operator
flag = True
print("Original", flag)
print("Reversed", not flag)

# Compare two floating-point
print(0.1 + 0.2 == 0.3)
print("Due to floating point precision errors")

# Take a user's age
age = int(input("Enter age "))
print("Not under 18", not(age < 18))

# Check if a number is positive
num = int(input("Enter number: "))
print("Positive and odd:", num > 0 and num % 2 != 0)

# Compare the lengths of two strings
a = input("Enter first string: ")
b = input("Enter second string: ")
print("First longer than second:", len(a) > len(b))
# ENDS HERE