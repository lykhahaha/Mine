# Matplotlib
## Line plot (1)  
![alt text](https://w2htrq.ch.files.1drv.com/y4mOVoV8jnGNsYrUvmAkhq5VcGuMNNx1cg-gVMHmxriSzHvlD9CwxazE8bb1FXedbniA-z_5ttrxuAar0h48AbHdmkwKRRsYtQ5BINBS17zqBT9ueXUm9qq0E9Bf4JmwyvsHuTNiy8x-CeMAFUYXQBhqQDwrCTXiaXB6d2sD7o_t_mINq020fbkaBeWLHyurvyINa5z7d-A7vj1X1idudmq6w/01.png?psid=1)
```python
# Print the last item from year and pop
print(year[-1])
print(pop[-1])

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year, pop)

# Display the plot with plt.show()
plt.show()
```

## Line plot (3)
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001219/02_gwsfiw.png)
```python
# Print the last item of gdp_cap and life_exp
print(gdp_cap[-1])
print(life_exp[-1])

# Make a line plot, gdp_cap on the x-axis, life_exp on the y-axis
plt.plot(gdp_cap, life_exp)

# Display the plot
plt.show()
```

## Scatter Plot (1)
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001219/03_zpzsro.png)
```python
# Change the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()
```

## Build a histogram (1)
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001218/04_srxght.png)
```python
# Create histogram of life_exp data
plt.hist(life_exp)

# Display histogram
plt.show()
```

## Build a histogram (2): bins
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001218/06_f8dsdd.png)
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001218/05_efzj0o.png)
```python
# Build histogram with 5 bins
plt.hist(life_exp, bins=5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(life_exp, bins=20)

# Show and clean up again
plt.show()
plt.clf()
```

## Build a histogram (3): compare
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001218/07_m6ym9q.png)
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001219/08_uldjjx.png)
```python
# Histogram of life_exp, 15 bins
plt.hist(life_exp, bins=15)

# Show and clear plot
plt.show()
plt.clf()

# Histogram of life_exp1950, 15 bins
plt.hist(life_exp1950, bins=15)

# Show and clear plot again
plt.show()
plt.clf()
```

## Labels
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001219/09_k7xbij.png)
```python
# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log') 

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)

# Add title
plt.title(title)

# After customizing, display the plot
plt.show()
```

## Ticks
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001219/10_vflkrw.png)
```python
# Scatter plot
plt.scatter(gdp_cap, life_exp)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')

# Definition of tick_val and tick_lab
tick_val = [1000,10000,100000]
tick_lab = ['1k','10k','100k']

# Adapt the ticks on the x-axis
plt.xticks(tick_val, tick_lab)

# After customizing, display the plot
plt.show()
```

## Sizes
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001219/11_dpnjxq.png)  
 `pop`: It contains population numbers for each country expressed in millions
```python
# Import numpy as np
import numpy as np

# Store pop as a numpy array: np_pop
np_pop = np.array(pop)

# Double np_pop
np_pop = np_pop * 2

# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s = np_pop)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

# Display the plot
plt.show()
```

## Colors
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001219/12_hqkg3d.png)
```python
# Specify c and alpha inside plt.scatter()
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha=0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Show the plot
plt.show()
```

## Additional Customizations
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519001220/13_jig3vy.png)
```python
# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()
```
# Dictionaries & Pandas
## Motivation for dictionaries
```python
# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# Get index of 'germany': ind_ger
ind_ger = countries.index('germany')

# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])
```

## Create dictionary
```python
# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# From string in countries and capitals, create dictionary europe
europe = {
    'spain':'madrid',
    'france':'paris',
    'germany':'berlin',
    'norway':'oslo'
}

# Print europe
print(europe)
```

## Access dictionary
```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe['norway'])
```

## Dictionary Manipulation (1)
```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Add italy to europe
europe['italy'] = 'rome'

# Print out italy in europe
print('italy' in europe)

# Add poland to europe
europe['poland'] = 'warsaw'

# Print europe
print(europe)
```

## Dictionary Manipulation (2)
```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn', 'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'australia':'vienna' }

# Update capital of germany
europe['germany'] = 'berlin'

# Remove australia
del(europe['australia'])

# Print europe
print(europe)
```

## Dictionariception
```python
# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } 
      	}


# Print out the capital of France
print(europe['france']['capital'])

# Create sub-dictionary data
data = {'capital':'rome', 'population':59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)
```

## Dictionary to DataFrame (1)
```python
# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {
    'country': names,
    'drives_right': dr,
    'cars_per_cap': cpc
}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)
```

## Dictionary to DataFrame (2)
```python
import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)
```

## CSV to DataFrame (1)
```python
# Import pandas as pd
import pandas as pd

# Import the cars.csv data: cars
cars = pd.read_csv('cars.csv')

# Print out cars
print(cars)
```

## CSV to DataFrame (2)
```python
# Import pandas as pd
import pandas as pd

# Fix import by including index_col
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out cars
print(cars)
```

## Square Brackets (1)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out country column as Pandas Series
print(cars['country'])

# Print out country column as Pandas DataFrame
print(cars[['country']])

# Print out DataFrame with country and drives_right columns
print(cars[['country', 'drives_right']])
```

## Square Brackets (2)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out first 3 observations
print(cars[0:3])

# Print out fourth, fifth and sixth observation
print(cars[3:6])
```

## loc and iloc (1)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out observation for Japan
print(cars.loc[['JAP']])

# Print out observations for Australia and Egypt
print(cars.loc[['AUS', 'EG']])
```

## loc and iloc (2)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out drives_right value of Morocco
print(cars.loc[['MOR'], ['drives_right']])

# Print sub-DataFrame
print(cars.loc[['RU', 'MOR'], ['country', 'drives_right']])
```

## loc and iloc (3)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out drives_right column as Series
print(cars.loc[:, 'drives_right'])

# Print out drives_right column as DataFrame
print(cars.loc[:, ['drives_right']])

# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:, ['cars_per_cap', 'drives_right']])
```
# Logic, Control Flow and Filtering
## Equality
```python
# Comparison of booleans
print(True == False)

# Comparison of integers
print(-5 * 15 != 75)

# Comparison of strings
print('pyscript' == 'PyScript')

# Compare a boolean with an integer
print(True == 1)
```

## Greater and less than
```python
# Comparison of integers
x = -3 * 6
print(x >= -10)

# Comparison of strings
y = "test"
print('test' <= y)

# Comparison of booleans
print(True > False)
```

## Compare arrays
```python
# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house[my_house >= 18])

# my_house less than your_house
print(my_house[my_house < your_house])
```
> [ 18.  20.]  
> [ 20.    10.75]

## and, or, not (1)
```python
# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
print(my_kitchen > 10 and my_kitchen < 18)

# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen < 14 or my_kitchen > 17)

# Double my_kitchen smaller than triple your_kitchen?
print(2 * my_kitchen < 3 * your_kitchen)
```

## Boolean operators with Numpy
```python
# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house > 18.5, my_house < 10))

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house < 11, your_house < 11))
```

## if
```python
# Define variables
room = "kit"
area = 14.0

# if statement for room
if room == "kit" :
    print("looking around in the kitchen.")

# if statement for area
if area > 15:
    print("big place!")
```

## Add else
```python
# Define variables
room = "kit"
area = 14.0

# if-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
else :
    print("looking around elsewhere.")

# if-else construct for area
if area > 15 :
    print("big place!")
else:
    print("pretty small.")
```

## Customize further: elif
```python
# Define variables
room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area > 10:
    print("medium size, nice!")
else :
    print("pretty small.")
```

## Driving right (1)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Extract drives_right column as Series: dr - a boolean Series,
dr = cars['drives_right']

# Use dr to subset cars: sel
sel = cars[dr]

# Print sel
print(sel)
```
> cars_per_cap&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;country&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;drives_right  
> US&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;809&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;United States&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True  
> RU&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;200&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Russia&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True  
> MOR&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;70&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Morocco&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True  
> EG&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;45&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Egypt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True  
## Driving right (2)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Convert code to a one-liner
sel = cars[cars['drives_right']]

# Print sel
print(sel)
```
## Cars per capita (1)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Create car_maniac: observations that have a cars_per_cap over 500
cpc = cars['cars_per_cap']
many_cars = cpc > 500
car_maniac = cars[many_cars]

# Print car_maniac
print(car_maniac)
```
> cars_per_cap&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;country&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;drives_right  
> US&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;809&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;United States&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True  
> AUS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;731&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Australia&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;False  
> JAP&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;588&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Japan&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;False  
## Cars per capita (2)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Import numpy, you'll need this
import numpy as np

# Create medium: observations with cars_per_cap between 100 and 500
medium = cars[np.logical_and(cars['cars_per_cap'] > 100, cars['cars_per_cap'] < 500)]

# Print medium
print(medium)
```
> cars_per_cap&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;country&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;drives_right  
> RU&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;200&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Russia&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True  
# Loops
## Basic while loop
```python
# Initialize offset
offset = 8

# Code the while loop
while offset != 0:
    print("correcting...")
    offset = offset - 1
    print(offset)
```

## Add conditionals
```python
# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0:
        offset = offset - 1
    else:
        offset = offset + 1
    print(offset)
```

## Loop over a list
```python
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for a in areas:
    print(a)
```

## Indexes and values (1)
```python
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate()
for index, height in enumerate(areas) :
    print("room " + str(index) + ": " + str(height))
```
> room 0: 11.25  
> room 1: 18.0  
> room 2: 20.0  
> room 3: 10.75  
> room 4: 9.5  
## Loop over list of lists
```python
# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch
for room in house:
    print("the " + room[0] + " is " + str(room[1]) + " sqm")
```
> the hallway is 11.25 sqm 
> the kitchen is 18.0 sqm 
> the living room is 20.0 sqm 
> the bedroom is 10.75 sqm 
> the bathroom is 9.5 sqm 
## Loop over dictionary
```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn', 
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'australia':'vienna' }
          
# Iterate over europe
for key, value in europe.items():
    print("the capital of " + key + " is " + value)
```
## Loop over Numpy array  
`np.nditer()`: to access the second column in 2D numpy array
```python
# Import numpy as np
import numpy as np

for x in np_baseball:
    print(x)

# For loop over np_baseball
for x in np.nditer(np_baseball):
    print(x)
```
> [ 75 190]  
> [ 73 195]  
> 190
> 195
## Loop over DataFrame (1)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for lab, row in cars.iterrows():
    print(lab)
    print(row)
```
> US  
> cars_per_cap&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;809  
> country&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;United States  
> drives_right&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True  
> Name: US, dtype: object  
## Loop over DataFrame (2)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab + ": " + str(row['cars_per_cap']))
```
> US: 809  
> AUS: 731  
> JAP: 588  
## Add column (1)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows():
    cars.loc[lab, 'COUNTRY'] = row['country'].upper()

# Print cars
print(cars)
```
> cars_per_cap&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;country&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;drives_right&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;COUNTRY  
> US&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;809&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;United States&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;UNITED STATES  
> AUS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;731&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Australia&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;False&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AUSTRALIA  
## Add column (2)
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Use .apply(str.upper)
for lab, row in cars.iterrows() :
    cars["COUNTRY"] = cars["country"].apply(str.upper)
    
print(cars)    
```
> cars_per_cap&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;country drives_right&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;COUNTRY  
> US&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;809&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;United States&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;UNITED STATES  
> AUS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;731&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Australia&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;False&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AUSTRALIA  
# Case Study: Hacker Statistics
## Random float
```python
# Import numpy as np
import numpy as np

# Set the seed
np.random.seed(123)

# Generate and print random float
print(np.random.rand())

np.random.seed(123)
print(np.random.rand())
```
> 0.6964691855978616  
> 0.6964691855978616  
## Roll the dice
```python
# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Use randint() to simulate a dice
print(np.random.randint(1, 7))

# Use randint() again
print(np.random.randint(1, 7))
```
> 6  
> 3  
## Determine your next move
```python
# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1, 7)

# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice <= 5 :
    step = step + 1
else :
    step = step + np.random.randint(1,7)

# Print out dice and step
print(dice)
print(step)
```
> 6  
> 53  
## The next step
```python
# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Initialize random_walk
random_walk = [0]

# Complete the ___
for x in range(100) :
    # Set step: last element in random_walk
    step = random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1,7)

    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)
```
> [0, 3, 4, 5, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, -1, 0, 5, 4, 3, 4, 3, 4, 5, 6, 7, 8, 7, 8, 7, 8, 9, 10, 11, 10, 14, 15, 14, 15, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 32, 33, 37, 38, 37, 38, 39, 38, 39, 40, 42, 43, 44, 43, 42, 43, 44, 43, 42, 43, 44, 46, 45, 44, 45, 44, 45, 46, 47, 49, 48, 49, 50, 51, 52, 53, 52, 51, 52, 51, 52, 53, 52, 55, 56, 57, 58, 57, 58, 59]
## How low can you go?
```python
# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Initialize random_walk
random_walk = [0]

for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        # Replace below: use max to make sure step can't go below 0
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

print(random_walk)
```
> [0, 3, 4, 5, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 6, 5, 4, 5, 4, 5, 6, 7, 8, 9, 8, 9, 8, 9, 10, 11, 12, 11, 15, 16, 15, 16, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 33, 34, 38, 39, 38, 39, 40, 39, 40, 41, 43, 44, 45, 44, 43, 44, 45, 44, 43, 44, 45, 47, 46, 45, 46, 45, 46, 47, 48, 50, 49, 50, 51, 52, 53, 54, 53, 52, 53, 52, 53, 54, 53, 56, 57, 58, 59, 58, 59, 60]
## Visualize the walk
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519006021/14_d0hmxd.png)
```python
# Initialization
import numpy as np
np.random.seed(123)
random_walk = [0]

for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()
```
## Simulate multiple walks
```python
# Initialization
import numpy as np
np.random.seed(123)

# Initialize all_walks
all_walks = []

# Simulate random walk 10 times
for i in range(10) :

    # Code from before
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)

    # Append random_walk to all_walks
    all_walks.append(random_walk)

# Print all_walks
print(all_walks)
```
> [[0, 3, 4, 5, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 6, 5, 4, 5, 4, 5, 6, 7, 8, 9, 8, 9, 8, 9, 10, 11, 12, 11, 15, 16, 15, 16, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 33, 34, 38, 39, 38, 39, 40, 39, 40, 41, 43, 44, 45, 44, 43, 44, 45, 44, 43, 44, 45, 47, 46, 45, 46, 45, 46, 47, 48, 50, 49, 50, 51, 52, 53, 54, 53, 52, 53, 52, 53, 54, 53, 56, 57, 58, 59, 58, 59, 60], [0, 4, 3, 2, 4, 3, 4, 6, 7, 8, 13, 12, 13, 14, 15, 16, 17, 16, 21, 22, 23, 24, 23, 22, 21, 20, 19, 20, 21, 22, 28, 27, 26, 25, 26, 27, 28, 27, 28, 29, 28, 33, 34, 33, 32, 31, 30, 31, 30, 29, 31, 32, 35, 36, 38, 39, 40, 41, 40, 39, 40, 41, 42, 43, 42, 43, 44, 45, 48, 49, 50, 49, 50, 49, 50, 51, 52, 56, 55, 54, 55, 56, 57, 56, 57, 56, 57, 59, 64, 63, 64, 65, 66, 67, 68, 69, 68, 69, 70, 71, 73], [0, 2, 1, 2, 3, 6, 5, 6, 5, 6, 7, 8, 7, 8, 7, 8, 9, 11, 10, 9, 10, 11, 10, 12, 13, 14, 15, 16, 17, 18, 17, 18, 19, 24, 25, 24, 23, 22, 21, 22, 23, 24, 29, 30, 29, 30, 31, 32, 33, 34, 35, 34, 33, 34, 33, 39, 38, 39, 38, 39, 38, 39, 43, 47, 49, 51, 50, 51, 53, 52, 58, 59, 61, 62, 61, 62, 63, 64, 63, 64, 65, 66, 68, 67, 66, 67, 73, 78, 77, 76, 80, 81, 82, 83, 85, 84, 85, 84, 85, 84, 83], [0, 6, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12, 13, 12, 11, 12, 11, 12, 11, 12, 13, 17, 18, 17, 23, 22, 21, 22, 21, 20, 21, 20, 24, 23, 24, 23, 24, 23, 24, 26, 25, 24, 23, 24, 23, 28, 29, 30, 29, 28, 29, 28, 29, 28, 33, 34, 33, 32, 31, 30, 31, 32, 36, 42, 43, 44, 45, 46, 45, 46, 48, 49, 50, 51, 50, 49, 50, 49, 50, 51, 52, 51, 52, 53, 54, 53, 52, 53, 54, 59, 60, 61, 66, 65, 66, 65, 66, 67, 68, 69, 68], [0, 6, 5, 6, 5, 4, 5, 9, 10, 11, 12, 13, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14, 13, 14, 15, 14, 15, 16, 19, 18, 19, 18, 19, 22, 23, 24, 25, 24, 23, 26, 27, 28, 29, 28, 27, 28, 31, 32, 37, 38, 37, 38, 37, 38, 37, 43, 42, 41, 42, 44, 43, 42, 41, 42, 43, 44, 45, 49, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 65, 64, 65, 66, 65, 71, 70, 71, 72, 71, 70, 71, 70, 69, 75, 74, 73, 74, 75, 74, 73], [0, 0, 0, 1, 7, 8, 11, 12, 18, 19, 20, 26, 25, 31, 30, 31, 32, 33, 32, 38, 39, 38, 39, 38, 39, 38, 39, 38, 39, 43, 44, 46, 45, 46, 45, 44, 45, 44, 45, 44, 48, 52, 51, 50, 49, 50, 51, 55, 56, 57, 61, 60, 59, 58, 59, 60, 62, 61, 60, 61, 62, 64, 67, 72, 73, 72, 73, 74, 75, 76, 77, 76, 77, 78, 84, 83, 88, 87, 91, 90, 94, 93, 96, 97, 96, 97, 103, 102, 101, 100, 104, 103, 102, 103, 104, 103, 104, 105, 106, 107, 106], [0, 0, 0, 1, 0, 0, 4, 5, 7, 11, 17, 16, 15, 16, 17, 18, 17, 18, 17, 18, 19, 18, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 33, 32, 35, 36, 35, 34, 35, 36, 37, 36, 35, 34, 33, 34, 35, 36, 37, 38, 39, 40, 39, 40, 41, 43, 42, 43, 44, 47, 49, 50, 49, 48, 47, 46, 45, 46, 45, 46, 48, 49, 50, 49, 50, 49, 48, 49, 48, 47, 46, 47, 46, 45, 46, 47, 48, 50, 51, 52, 51, 50, 51, 57, 56, 57, 58, 63, 62, 63], [0, 0, 1, 2, 1, 2, 3, 9, 10, 11, 12, 11, 13, 14, 15, 16, 15, 16, 17, 18, 19, 18, 19, 18, 19, 20, 19, 20, 24, 25, 28, 29, 33, 34, 33, 34, 35, 34, 33, 38, 39, 40, 39, 38, 39, 40, 41, 40, 44, 43, 44, 45, 46, 47, 48, 49, 50, 49, 48, 47, 48, 49, 53, 54, 53, 54, 55, 54, 60, 61, 62, 63, 62, 63, 64, 67, 66, 67, 66, 65, 64, 65, 66, 68, 69, 70, 74, 75, 74, 73, 74, 75, 74, 73, 74, 75, 76, 75, 74, 75, 76], [0, 1, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 5, 10, 14, 13, 14, 13, 12, 11, 12, 11, 12, 13, 12, 16, 17, 16, 17, 16, 15, 16, 15, 19, 20, 21, 22, 23, 24, 23, 24, 25, 26, 27, 28, 27, 32, 33, 34, 33, 34, 33, 34, 35, 34, 35, 40, 41, 42, 41, 42, 43, 44, 43, 44, 43, 44, 45, 44, 43, 42, 43, 44, 43, 42, 41, 42, 46, 47, 48, 49, 50, 51, 50, 51, 52, 51, 52, 57, 58, 57, 56, 57, 56, 55, 54, 58, 59, 60, 61, 60], [0, 1, 2, 3, 4, 5, 4, 3, 6, 5, 4, 3, 2, 3, 9, 10, 9, 10, 11, 10, 9, 10, 11, 12, 11, 15, 16, 15, 17, 18, 17, 18, 19, 20, 21, 22, 23, 22, 21, 22, 23, 22, 23, 24, 23, 22, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 33, 34, 35, 36, 37, 38, 37, 36, 42, 43, 44, 43, 42, 41, 45, 46, 50, 49, 55, 56, 57, 61, 62, 61, 60, 61, 62, 63, 64, 63, 69, 70, 69, 73, 74, 73, 74, 73, 79, 85, 86, 85, 86, 87]]

## Visualize all walks
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519006466/15_eljjr0.png)
```python
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
all_walks = []
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to Numpy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()

# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()
```

## Plot the distribution
![alt text](http://res.cloudinary.com/dqagyeboj/image/upload/v1519006655/16_figipa.png)
```python
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
all_walks = []

# Simulate random walk 500 times
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
ends = np.array(np_aw_t[-1])

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()
```