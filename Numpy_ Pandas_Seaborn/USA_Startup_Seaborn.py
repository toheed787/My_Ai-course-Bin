import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#https://seaborn.pydata.org/generated/seaborn.set_theme.html
#https://seaborn.pydata.org/tutorial/aesthetics.html
#https://python-charts.com/seaborn/themes/

# Sample data
data = pd.DataFrame({'x': np.arange(100), 'y': np.random.rand(100).cumsum()})

# Set the theme
sns.set_theme(style='darkgrid')
# Alternatively
# sns.set_style('darkgrid')

# Create a plot
sns.lineplot(x='x', y='y', data=data)
plt.show()

# Other themes can be set similarly
sns.set_theme(style='whitegrid')
sns.lineplot(x='x', y='y', data=data)
plt.show()

sns.set_theme(style='dark')
sns.lineplot(x='x', y='y', data=data)
plt.show()

sns.set_theme(style='white')
sns.lineplot(x='x', y='y', data=data)
plt.show()

sns.set_theme(style='ticks')
sns.lineplot(x='x', y='y', data=data)
plt.show()


"""Customizing Themes
It is possible to customize the themes further by passing a dictionary of parameters to the rc argument of seaborn.set_theme() or seaborn.set_style(). This allows for fine-grained control over the appearance of plots."""

# Customize the theme
sns.set_theme(style='darkgrid', rc={'axes.facecolor': 'grey', 'grid.color': 'white'})

# Create a plot
sns.lineplot(x='x', y='y', data=data)
plt.show()

#Zameencom data - based examples
# Load data from a CSV file
df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Numpy_ Pandas_Seaborn\USA_startup_csv.csv',delimiter=",", index_col='Startup Name')

print(df.dtypes)
dffilter= df.head(40)
dffilter100= df.head(100)
# https://seaborn.pydata.org/api.html#distribution-api


#https://seaborn.pydata.org/generated/seaborn.set_theme.html
#https://seaborn.pydata.org/tutorial/aesthetics.html
#https://seaborn.pydata.org/tutorial/color_palettes.html

sns.set(style="whitegrid")


#kind='hist'  
g=sns.displot(data=dffilter, x="Industry" , y="Investment Amount (USD)" , hue="Country",  kind='hist'  )
g.figure.suptitle("sns.displot(data=dffilter, x=Industry , y=Investment Amount (USD) , hue=Country,  kind='hist'  )"  )

# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()


""""kind="kde" in Seaborn specifies the use of Kernel Density Estimation plots. KDE plots visualize the probability density of a continuous variable. Instead of discrete bins like in histograms, KDE plots use a continuous curve to estimate the underlying distribution of the data. This provides a smoother and often more informative representation of the data's distribution, especially for continuous variables."""
#kind='kde'
g=sns.displot(data=dffilter, x="Investment Amount (USD)" , y="Year Founded" , kind='kde'  )
g.figure.suptitle("sns.displot(data=dffilter, x=Investment Amount (USD) , y=Year Founded, kind='kde'  )"  )

# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()

#https://seaborn.pydata.org/generated/seaborn.kdeplot.html
#kind='kde'
g=sns.kdeplot(data=dffilter, x="Investment Amount (USD)")
g.figure.suptitle("sns.kdeplot(data=dffilter, x=Investment Amount (USD))"  )

# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()


# See: https://seaborn.pydata.org/generated/seaborn.histplot.html#seaborn.histplot
"""Plot univariate or bivariate histograms to show distributions of datasets.
A histogram is a classic visualization tool that represents the distribution of one or more variables by counting the number of observations that fall within discrete bins."""
g = sns.histplot(data=dffilter, x='Industry', y='Investment Amount (USD)', hue='Industry', multiple="stack")
g.figure.suptitle("sns.histplot(data=dffilter, x='Industry', y='Investment Amount (USD)', hue='Industry', multiple=stack)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()

#https://seaborn.pydata.org/generated/seaborn.scatterplot.html#seaborn.scatterplot
"""Draw a scatter plot with possibility of several semantic groupings.

The relationship between x and y can be shown for different subsets of the data using the hue, size, and style parameters. These parameters control what visual semantics are used to identify the different subsets. It is possible to show up to three dimensions independently by using all three semantic types, but this style of plot can be hard to interpret and is often ineffective. Using redundant semantics (i.e. both hue and style for the same variable) can be helpful for making graphics more accessible."""
# Use Seaborn to create a plot
g = sns.scatterplot(x='Industry', y='Investment Amount (USD)', data=dffilter)
g.figure.suptitle("sns.scatterplot(x='Industry', y='Investment Amount (USD)', data=dffilter)"  )
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()


#https://seaborn.pydata.org/generated/seaborn.lineplot.html
"""Draw a line plot with possibility of several semantic groupings.

The relationship between x and y can be shown for different subsets of the data using the hue, size, and style parameters. These parameters control what visual semantics are used to identify the different subsets. It is possible to show up to three dimensions independently by using all three semantic types, but this style of plot can be hard to interpret and is often ineffective. Using redundant semantics (i.e. both hue and style for the same variable) can be helpful for making graphics more accessible."""
g=sns.lineplot(data=dffilter, x="Industry" , y="Investment Amount (USD)"  )
g.figure.suptitle("sns.lineplot(data=dffilter, x=Industry , y=Investment Amount (USD) )"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()



#https://seaborn.pydata.org/generated/seaborn.barplot.html
"""Show point estimates and errors as rectangular bars.

A bar plot represents an aggregate or statistical estimate for a numeric variable with the height of each rectangle and indicates the uncertainty around that estimate using an error bar. Bar plots include 0 in the axis range, and they are a good choice when 0 is a meaningful value for the variable to take."""
g=sns.barplot(data=dffilter, x="Industry", y="Investment Amount (USD)", legend=False)
g.figure.suptitle("sns.barplot(data=dffilter, x=Industry, y=Investment Amount (USD), legend=False)"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")
#g.figure.clear()


#https://seaborn.pydata.org/generated/seaborn.catplot.html
""""Figure-level interface for drawing categorical plots onto a FacetGrid.

This function provides access to several axes-level functions that show the relationship between a numerical and one or more categorical variables using one of several visual representations. The kind parameter selects the underlying axes-level function to use."""

g=sns.catplot(data=dffilter, x="Industry", y="Investment Amount (USD)")
g.figure.suptitle("sns.catplot(data=df, x=Industry, y=Investment Amount (USD))"  )
# Display the plot
g.figure.show() 
read = input("Wait for me....")
#g.figure.clear()




#https://seaborn.pydata.org/generated/seaborn.heatmap.html
""""Plot rectangular data as a color-encoded matrix.

This is an Axes-level function and will draw the heatmap into the currently-active Axes if none is provided to the ax argument. Part of this Axes space will be taken and used to plot a colormap, unless cbar is False or a separate Axes is provided to cbar_ax."""
#.pivot(index="Model", columns="agency", values="price")
glue = dffilter.pivot(columns="Industry", values="Investment Amount (USD)")

g=sns.heatmap(glue)
g.figure.suptitle("sns.heatmap(glue)  - glue = dffilter.pivot(columns=Industry, values=Investment Amount (USD))"  )
# Display the plot
g.figure.show()
read = input("Wait for me....")