# Pytthon Toolbox 2
## Summary
* Iterate for
	* List `flash`: `it = iter(flash)`
	* `range()`: `it = iter(range(3))`
	* `enumerate(flash, start=1)`: `it = iter(enumerate(flash, start=1))`
	> (1, 'charles xavier')
	* `zip(mutants, aliases)`: `it = iter(zip(mutants, aliases))`
	* Generator (cannot `*gen_fellow`): `gen_fellow = (member for member in fellowship)`, it's different with `gen_fellow = (1, 2, 3)` because `gen_follow` here is `tuple`
	* `chunk`: 
	```python

		urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

		df_urb_pop = next(urb_pop_reader)

		print(df_urb_pop.head())
	```
* Access iterator `it`: `print(next(it))`: access each value or `print(*it)`: access all values
* Unzip `z1 = zip(mutants, powers)`: `result1, result2 = zip(*z1)`
* Sum the list `values`: `sum(values)`
* Usage of `chunk`
```python
for chunk in pd.read_csv('tweets.csv', chunksize=10):
	for entry in chunk['lang']
```
* List Comprehensions:
	* Nested list (matrix 5 x 5): `[[col for col in range(5)] for row in range(5)]`
	* Condition `if`:
		* `new_fellowship = [member for member in fellowship if len(member) > 6]`
		* `new_fellowship = [member if len(member) > 6 else '' for member in fellowship]`
* Dict comprehensions: `len_fellow = {member: len(member) for member in fellowship}`
* Build a generator thru function:
```python
def gen_fellow(input_list):
    """Generator that yields length of input_list"""
    for member in input_list:
        yield len(member)
        
gen_instance = gen_fellow(fellowship)
print(next(gen_instance))
```
* `DataFrame` the `dict(zip)`
* Read each line in `with open('example.csv') as file`
```python
while True:
	line = file.readline()
	if not line:
		break
	yield line
```
* `DataFrame.plot(kind = 'scatter', x = '', y = '')` is more accurate than `plt.scatter(x = DataFrame[], y = DataFrame[])`
## Using iterators in PythonLand
### Iterators vs Iterables
> flash1  
> ['jay garrick', 'barry allen', 'wally west', 'bart allen']  
> flash2  
> <list_iterator at 0x7f45281f11d0>  
> flash1 is an iterable and flash2 is an iterator.
### Iterating over iterables (1)
```python
# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# Print each list item in flash using a for loop
for person in flash:
    print(person)

# Create an iterator for flash: superspeed
it = iter(flash)

# Print each item from the iterator
print(*it)
print(*it)
print(*it)
print(*it)
```
> jay garrick  
> barry allen  
> wally west  
> bart allen  
> jay garrick barry allen wally west bart allen
### Iterating over iterables (2)
Recall that `range()` doesn't actually create the list; instead, it creates a range object with an iterator that produces the values until it reaches the limit (in the example, until the value 4). If `range()` created the actual list, calling it with a value of 10<sup>100</sup> may not work, especially since a number as big as that may go over a regular computer's memory. The value 10<sup>100</sup> is actually what's called a Googol which is a 1 followed by a hundred 0s. That's a huge number!  
Your task for this exercise is to show that calling range() with 10<sup>100</sup> won't actually pre-create the list.
```python
# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for num in range(3):
    print(num)

# Create an iterator for range(10 ** 100): googol
googol = iter(range(10**100))

# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
```
### Iterators as function arguments
There are also functions that take iterators as arguments. For example, the `list()` and `sum()` functions return a list and the sum of elements, respectively. 
```python
# Create a range object: values
values = range(10, 21)

# Print the range object
print(values)

# Create a list of integers: values_list
values_list = list(values)

# Print values_list
print(values_list)

# Get the sum of values: values_sum
values_sum = sum(values)

# Print values_sum
print(values_sum)
``` 
> range(10, 21)  
> [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  
> 165
### Using enumerate
`enumerate()` returns an `enumerate` object that produces a sequence of tuples, and each of the tuples is an _index-value_ pair.
```python
# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pride']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2, value2 in enumerate(mutants, start=1):
    print(index2, value2)
```
> [(0, 'charles xavier'), (1, 'bobby drake'), (2, 'kurt wagner'), (3, 'max eisenhardt'), (4, 'kitty pride')]  
> 0 charles xavier  
> 1 bobby drake  
> 2 kurt wagner  
> 3 max eisenhardt  
> 4 kitty pride  
> 1 charles xavier  
> 2 bobby drake  
> 3 kurt wagner  
> 4 max eisenhardt  
> 5 kitty pride
### Using zip
If you wanted to print the values of a `zip` object, you can convert it into a list and then print it. Printing just a `zip` object will not return the values unless you unpack it first  
```python
# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))

# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
for value1, value2, value3 in zip(mutants, aliases, powers):
    print(value1, value2, value3)
```
> [('charles xavier', 'prof x', 'telepathy'), ('bobby drake', 'iceman', 'thermokinesis'), ('kurt wagner', 'nightcrawler', 'teleportation'), ('max eisenhardt', 'magneto', 'magnetokinesis'), ('kitty pride', 'shadowcat', 'intangibility')]  
> <zip object at 0x7f452818a148>  
> charles xavier prof x telepathy  
> bobby drake iceman thermokinesis  
> kurt wagner nightcrawler teleportation  
> max eisenhardt magneto magnetokinesis  
> kitty pride shadowcat intangibility
### Using * and zip to 'unzip'
```python
# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the tuples in z1 by unpacking with *
print(*z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)
```
> ('charles xavier', 'telepathy') ('bobby drake', 'thermokinesis') ('kurt wagner', 'teleportation') ('max eisenhardt', 'magnetokinesis') ('kitty pride', 'intangibility')  
> True  
> True
### Processing large amounts of Twitter data
Sometimes, the data we have to process reaches a size that is too much for a computer's memory to handle   
```python
# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('tweets.csv', chunksize=10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)
```
> {'und': 2, 'en': 97, 'et': 1}
### Extracting information for large amounts of Twitter data
It's good to know how to process a file in smaller, more manageable chunks, but it can become very tedious having to write and rewrite the same code for the same task each time.  
```python
# Define count_entries()
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize=c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict

# Call count_entries(): result_counts
result_counts = count_entries('tweets.csv', 10, 'lang')

# Print result_counts
print(result_counts)
```
> {'und': 2, 'en': 97, 'et': 1}
## List comprehensions and generators
### Write a basic list comprehension
> doctor = ['house', 'cuddy', 'chase', 'thirteen', 'wilson']  
> The list comprehension is `[doc[0] for doc in doctor]` and produces the list `['h', 'c', 'c', 't', 'w']`.
### List comprehension over iterables
> doctor = ['house', 'cuddy', 'chase', 'thirteen', 'wilson']  
>   
> range(50)  
>   
> underwood = 'After all, we are nothing more or less than what we choose to reveal.'  
>   
> jean = '24601'  
>   
> flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']  
>   
> valjean = 24601  
> You can build list comprehensions over all the objects except the integer object `valjean`.
### Writing list comprehensions
```python
squares = [i * i for i in range(0, 9)]
```
### Nested list comprehensions
Matrices can be represented as a list of lists in Python.  
To create the list of lists, you simply have to supply the list comprehension as the output expression of the overall list comprehension:`[[output expression] for iterator variable in iterable]`
```python
# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

# Print the matrix
for row in matrix:
    print(row)
```
> [0, 1, 2, 3, 4]  
> [0, 1, 2, 3, 4]  
> [0, 1, 2, 3, 4]  
> [0, 1, 2, 3, 4]  
> [0, 1, 2, 3, 4]
### Using conditionals in comprehensions (1)
`[ output expression for iterator variable in iterable if predicate expression ].`
```python
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member for member in fellowship if len(member) > 6]

# Print the new list
print(new_fellowship)
```
> ['samwise', 'aragorn', 'legolas', 'boromir']
### Using conditionals in comprehensions (2)
```python
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member if len(member) > 6 else '' for member in fellowship]

# Print the new list
print(new_fellowship)
```
> ['', 'samwise', '', 'aragorn', 'legolas', 'boromir', '']
### Dict comprehensions
```python
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create dict comprehension: new_fellowship
new_fellowship = {member: len(member) for member in fellowship}

# Print the new list
print(new_fellowship)
```
> {'gimli': 5, 'frodo': 5, 'boromir': 7, 'aragorn': 7, 'samwise': 7, 'legolas': 7, 'merry': 5}
### List comprehensions vs generators
```python
# List of strings
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# List comprehension
fellow1 = [member for member in fellowship if len(member) >= 7]

# Generator expression
fellow2 = (member for member in fellowship if len(member) >= 7)
```
> A list comprehension produces a list as output, a generator produces a generator object.
### Write your own generator expressions
Recall that generator expressions basically have the same syntax as list comprehensions, except that it uses parentheses `()` instead of brackets `[]`; this should make things feel familiar! Furthermore, if you have ever iterated over a dictionary with `.items()`, or used the `range()` function, for example, you have already encountered and used generators before, without knowing it! When you use these functions, Python creates generators for you behind the scenes.
```python
# Create generator object: result
result = (num for num in range(31))

# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))

# Print the rest of the values
for value in result:
    print(value)
```
### Changing the output in generator expressions
```python
# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Create a generator object: lengths
lengths = (len(person) for person in lannister)

# Iterate over and print the values in lengths
for value in lengths:
    print(value)
```
> 6  
> 5  
> 5  
> 6  
> 7
### Build a generator
```python
# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Define generator function get_lengths
def get_lengths(input_list):
    """Generator function that yields the
    length of the strings in input_list."""

    # Yield the length of a string
    for person in input_list:
        yield len(person)

# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)
```
> 6  
> 5  
> 5  
> 6  
> 7
### List comprehensions for time-stamped data
```python
# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time]

# Print the extracted times
print(tweet_clock_time)
```
> ['23:40:17', '23:40:17', '23:40:17', '23:40:17', '23:40:17', '23:40:17', '23:40:18', '23:40:17']
### Conditional list comprehesions for time-stamped data
```python
# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time if entry[17:19] == '19']

# Print the extracted times
print(tweet_clock_time)
```
> ['23:40:19', '23:40:19', '23:40:19', '23:40:19', '23:40:19', '23:40:19', '23:40:19']
## Bringing it all together!
### Dictionaries for data science ([source](http://data.worldbank.org/data-catalog/world-development-indicators))
```python
# Zip lists: zipped_lists
zipped_lists = zip(feature_names, row_vals)

# Create a dictionary: rs_dict
rs_dict = dict(zipped_lists)

# Print the dictionary
print(rs_dict)
```
> {'IndicatorCode': 'SP.ADO.TFRT', 'Year': '1960', 'CountryName': 'Arab World', 'CountryCode': 'ARB', 'IndicatorName': 'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'Value': '133.56090740552298'}
### Writing a function to help you
```python
# Define lists2dict()
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(list1, list2)

    # Create a dictionary: rs_dict
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return rs_dict

# Call lists2dict: rs_fxn
rs_fxn = lists2dict(feature_names, row_vals)

# Print rs_fxn
print(rs_fxn)
```
> {'IndicatorCode': 'SP.ADO.TFRT', 'Year': '1960', 'CountryName': 'Arab World', 'CountryCode': 'ARB', 'IndicatorName': 'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'Value': '133.56090740552298'}
### Using a list comprehension
```python
# Print the first two lists in row_lists
print(row_lists[0])
print(row_lists[1])

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]

# Print the first two dictionaries in list_of_dicts
print(list_of_dicts[0])
print(list_of_dicts[1])
```
> ['Arab World', 'ARB', 'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'SP.ADO.TFRT', '1960', '133.56090740552298']  
> ['Arab World', 'ARB', 'Age dependency ratio (% of working-age population)', 'SP.POP.DPND', '1960', '87.7976011532547']  
> {'CountryName': 'Arab World', 'IndicatorName': 'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'IndicatorCode': 'SP.ADO.TFRT', 'CountryCode': 'ARB', 'Year': '1960', 'Value': '133.56090740552298'}  
> {'CountryName': 'Arab World', 'IndicatorName': 'Age dependency ratio (% of working-age population)', 'IndicatorCode': 'SP.POP.DPND', 'CountryCode': 'ARB', 'Year': '1960', 'Value': '87.7976011532547'}
### Turning this all into a DataFrame
```python
# Import the pandas package
import pandas as pd

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]

# Turn list of dicts into a DataFrame: df
df = pd.DataFrame(list_of_dicts)

# Print the head of the DataFrame
print(df.head())
```
> CountryCode&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CountryName&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IndicatorCode&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IndicatorName&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Value&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Year  
> 0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ARB&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Arab World&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SP.ADO.TFRT&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Adolescent fertility rate (births per 1,000 wo...  133.56090740552298&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1960  
> 1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ARB&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Arab World&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SP.POP.DPND&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Age dependency ratio (% of working-age populat...    87.7976011532547&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1960  
> 2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ARB&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Arab World&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SP.POP.DPND.OL&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Age dependency ratio, old (% of working-age po...   6.634579191565161&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1960
### Processing data in chunks (1)
```python
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Skip the column names
    file.readline()

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Process only the first 1000 rows
    for j in range(1000):

        # Split the current line into a list: line
        line = file.readline().split(',')

        # Get the value for the first column: first_col
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1

# Print the resulting dictionary
print(counts_dict)
```
> {'European Union': 116, 'Central Europe and the Baltics': 71, 'Europe & Central Asia (developing only)': 89, 'East Asia & Pacific (all income levels)': 122, 'Arab World': 80, 'East Asia & Pacific (developing only)': 123, 'Europe & Central Asia (all income levels)': 109, 'Caribbean small states': 77, 'Heavily indebted poor countries (HIPC)': 18, 'Fragile and conflict affected situations': 76, 'Euro area': 119}

### Writing a generator to load data in chunks (2)
In this case, it would be useful to use generators. Generators allow users to [lazily evaluate data](http://www.blog.pythonlibrary.org/2014/01/27/python-201-an-intro-to-generators/). This concept of lazy evaluation is useful when you have to deal with very large datasets because it lets you generate values in an efficient manner by yielding only chunks of data at a time instead of the whole thing at once.  
```python
# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""

    # Loop indefinitely until the end of the file
    while True:

        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data
        
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))
```
> CountryName,CountryCode,IndicatorName,IndicatorCode,Year,Value  
> Arab World,ARB,"Adolescent fertility rate (births per 1,000 women ages 15-19)",SP.ADO.TFRT,1960,133.56090740552298  
> Arab World,ARB,Age dependency ratio (% of working-age population),SP.POP.DPND,1960,87.7976011532547

### Writing a generator to load data in chunks (3)
```python
# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Iterate over the generator from read_large_file()
    for line in read_large_file(file):

        row = line.split(',')
        first_col = row[0]

        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1

# Print            
print(counts_dict)
```
> {'Fragile and conflict affected situations': 76, 'Low & middle income': 138, 'High income: OECD': 127, 'Pacific island small states': 66, 'Small states': 69, 'OECD members': 130, 'High income': 131, 'North America': 123, 'Lower middle income': 126, 'High income: nonOECD': 68, 'Latin America & Caribbean (developing only)': 133, 'Other small states': 63, 'Euro area': 119, 'East Asia & Pacific (developing only)': 123, 'Middle income': 138, 'Least developed countries: UN classification': 78, 'CountryName': 1, 'Heavily indebted poor countries (HIPC)': 99, 'Europe & Central Asia (developing only)': 89, 'Latin America & Caribbean (all income levels)': 130, 'Arab World': 80, 'Middle East & North Africa (all income levels)': 89, 'Middle East & North Africa (developing only)': 94, 'European Union': 116, 'Central Europe and the Baltics': 71, 'East Asia & Pacific (all income levels)': 122, 'South Asia': 36, 'Low income': 80, 'Europe & Central Asia (all income levels)': 109, 'Caribbean small states': 77}
### Writing an iterator to load data in chunks (1)
```python
# Import the pandas package
import pandas as pd

# Initialize reader object: df_reader
df_reader = pd.read_csv('ind_pop.csv', chunksize=10)

# Print two chunks
print(next(df_reader))
print(next(df_reader))
```
> CountryName&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CountryCode&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IndicatorName&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IndicatorCode&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Year&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Value  
> 0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Arab World&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ARB&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Caribbean small states&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CSS  
> CountryName&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CountryCode&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IndicatorName&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IndicatorCode&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Year&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Value  
> 0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Arab World&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ARB&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Caribbean small states&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CSS 

### Writing an iterator to load data in chunks (2)
```python
# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out the head of the DataFrame
print(df_urb_pop.head())

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'], df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Print pops_list
print(pops_list)
```
> CountryName&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CountryCode&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Year&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Total Population&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Urban population (% of total)  
> Arab World&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ARB&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1960&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.249590e+07&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;31.285384  
> Caribbean small states&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CSS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1960&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.190810e+06&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;31.597490  
> [(91401583.0, 44.507921139002597), (92237118.0, 45.206665319194002), (93014890.0, 45.866564696018003), (93845749.0, 46.5340927663649), (94722599.0, 47.208742980352604)]

### Writing an iterator to load data in chunks (3)
![alt text](https://woypoa.ch.files.1drv.com/y4m8ebrQ2wAUaL2ZWEBwt5YZjLArWCfa7J_EVBy0FJ9KEVq40B9Ana9BgueTW-pR6ZwH377bj86xalSDLP9hJDiKv_Im_dQDB6QdjmfT5z-2X8mtVRbAlPqNcL-kf5lkBipH_b-Ky8uFH8WEqQs50zfNjV0YGaXqYKxYuKU29xzvEg-vlE0EgDmzq5bc8MOt2zXa_qqVSTX1IiQE5OlgWA2JA/16.PNG?psid=1)
```python
# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'], 
            df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Use list comprehension to create new DataFrame column 'Total Urban Population'
df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]

# Plot urban population data
df_pop_ceb.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()
```

### Writing an iterator to load data in chunks (4)
![alt text](https://wowjxa.ch.files.1drv.com/y4mBOrx4Tyv7-Ro4tZJ-irI18qgdiFEK16jFmRGHJooV80ghQkLoelBVS00d1VuCptz7ZUqSIo4HOQGpNcN-WkKKWdB7l70Lt0wiSoLBtz7HATHG3m9QEvMIEdRew7ujg9Sbu49FLR41z56qSYKa9MYz5bVLso8yeck8hRCdaVBhX56IQniUchoYQPUuUffVzG6CIqFLD3Swsei7vo1RIn5gQ/17.PNG?psid=1)
```python
# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Initialize empty DataFrame: data
data = pd.DataFrame()

# Iterate over each DataFrame chunk
for df_urb_pop in urb_pop_reader:

    # Check out specific country: df_pop_ceb
    df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

    # Zip DataFrame columns of interest: pops
    pops = zip(df_pop_ceb['Total Population'],
                df_pop_ceb['Urban population (% of total)'])

    # Turn zip object into list: pops_list
    pops_list = list(pops)

    # Use list comprehension to create new DataFrame column 'Total Urban Population'
    df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1]) for tup in pops_list]
    
    # Append DataFrame chunk to data: data
    data = data.append(df_pop_ceb)

# Plot urban population data
data.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()
```

### Make a function for above code
![alt text](https://wowjxa.ch.files.1drv.com/y4mBOrx4Tyv7-Ro4tZJ-irI18qgdiFEK16jFmRGHJooV80ghQkLoelBVS00d1VuCptz7ZUqSIo4HOQGpNcN-WkKKWdB7l70Lt0wiSoLBtz7HATHG3m9QEvMIEdRew7ujg9Sbu49FLR41z56qSYKa9MYz5bVLso8yeck8hRCdaVBhX56IQniUchoYQPUuUffVzG6CIqFLD3Swsei7vo1RIn5gQ/17.PNG?psid=1)
![alt text](https://wowcfq.ch.files.1drv.com/y4msDfOfSuc6qFnRUfHN3zkL4b4PZjbanGXjZL9eoqoGOCAThVEO-0XrEeuZ2za5aWD3ip4Xr-RQV7XIrEaWS8OX4_zl2vBe6jGbBgVSVCZHF_Sro1TTUfQTQKg9qYxaeBR3qDcDzre9n5S3J4T3Qh9uET-tlpiGtztES_KRKsrAnika6hBeD4WEJ-WKSCHqynxQVvA-f3fX3rRD_Bhp9BxOw/18.PNG?psid=1)
```python
# Define plot_pop()
def plot_pop(filename, country_code):

    # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    # Initialize empty DataFrame: data
    data = pd.DataFrame()
    
    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
        # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                    df_pop_ceb['Urban population (% of total)'])

        # Turn zip object into list: pops_list
        pops_list = list(pops)

        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1]) for tup in pops_list]
    
        # Append DataFrame chunk to data: data
        data = data.append(df_pop_ceb)

    # Plot urban population data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()

# Set the filename: fn
fn = 'ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop(fn, 'CEB')

# Call plot_pop for country code 'ARB'
plot_pop(fn, 'ARB')
```