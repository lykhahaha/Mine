# Import Data
## Keyword
* `open('', mode='r')`,`with open('') as file`, `file.readline()`, `file.close()`
* `
## Summary
* Use `open`
	* `open` just read the file or each line in file, without `delimiter` or `sep`
	* Step 1: `file = open('moby_dick.txt', mode='r')` or with context manager: `with open('moby_dick.txt') as file:`
	* Step 2: `file.read()` or `file.readline()`
	* Step 3: `file.close()`
* Use `NumPy` to import
	* `digits = np.loadtxt('figits.csv', delimiter=',')`
		* This `digits` is in `numpy.ndarray`
		* `delimiter` can be `\t`, `,`
		* `skiprows` need to be **1** for first row
		* `usecols=[0, 2]`
		* Dataset is likely to have label in the header, so to avoid `could not convert string to float: label`, 2 solutions:
			* Set `dtype` to `str`: `digits = np.loadtxt('digits_header.txt', delimiter='\t', dtype=str, usecols=[0, 2])`
			* Skip first row by `skiprows = 1`: `digits = np.loadtxt('digits_header.txt', delimiter='\t', skiprows=1, usecols=[0, 2])`
		* Result from `np.loadtxt()`:
		> [['1' '0' '3' 'male' '22.0' '1' '0' 'A/5 21171']]

	* `np.genfromtxt()` can solve mixed datatypes datasets
		* `data = np.genfromtxt('seaslug.txt', delimiter='\t', names=True, dtype=None)`
		* `names: True`: Dataset has label so `genfromtxt()` will skip this line
		* if dataset is mixed datatype, `np.genfromtxt()` will generate all element be `str`
		* Result from `np.genfromtxt()`:
		> [(1, 0, 3, b'male', 22., 1, 0, b'A/5 21171')]

	* `np.recfromcsv()` is more specific to `csv`
		* `data = np.recfromcsv('titanic.csv', delimiter=',', names=True, dtype=None)`
		* Result from `np.recfromcsv()`
		> [(1, 0, 3, b'male', 22., 1, 0, b'A/5 21171')]
* Use `pandas` to import
```python
# Read the first 5 rows of the file into a DataFrame: data
data = pd.read_csv(file, nrows=5, header=None)

# Build a numpy array from the DataFrame: data_array
data_array = data.values
```
> [['1' '0' '3' 'male' '22.0' '1' '0' 'A/5 21171']]  

* Aside from csv, `read_csv` can load `.txt` which contains `\t` as separator, some comments with `#` and missing values with `NA/NaN/Nothing/?`
```python
data = pd.reav_csv('titanic_corrput.txt', sep='\t', comments='#', na_values=['NA', 'NaN', 'Nothing'])
```

* View directory by `os`
```python
import os
wd = os.getcwd()
os.listdir(wd)
```
* Use `pickle` to load binary file
	* The result is `dict`
```python
# Import pickle package
import pickle

# Open pickle file and load data: d
with open('data.pkl', 'rb') as file:
    d = pickle.load(file)
```
> {'Aug': '85', 'Mar': '84.4', 'Airline': '8', 'June': '69.4'}

* Load Excel file by `pandas`
	* `data = pd.ExcelFile('battledeath.xlsx')`
	* `print(data.sheet_names)` to print sheet name in Excel file
	* `df1 = data.parse('2004')` or `df1 = data.parse(0)` to access to every sheet and `print(df1.head())`
	* `data1 = data.parse('2004', skiprows=1, names=['Country', 'AAM due to War (2002)'])` to skip the first row and name the label/first row
	* `data1 = data.parse('2004', parse_cols=0, skiprows=1, names=['Country'])`: skip the first row, just get first attribute through `parse_cols=0` and name it, `parse_col` just works in Python 2

* Load SAS File by `SAS7BDAT`
	* `from sas7bdat import SAS7BDAT`
	* `with SAS7BDAT('sales.sas7bdat') as file: df_sas = file.to_data_frame()`
	* Aside from `plt.hist(df_sas['P'])` to plot `P` attribute, we can `pd.DataFrame.hist(df_sas[['P']])`

* Load Stata file by `pd.read_stata`: `df = pd.read_stata('disarea.dta')`

* Load HDF5 file (Big file)
	* `data = h5py.File(file, 'r')`
	* `strain = data['strain']['Strain'].value`
	* The result from `strain` is `[2.17704028e-19 2.08763900e-19 2.39681183e-19 2.28672230e-19
 2.13224038e-19]`

* Load Matlab file
	* `mat = scipy.io.loadmat('data2.mat')` and mat will be in `dict`
	* `print(mat.keys())` to get the keys
	* `data = mat['CYratioCyt'][25, 5:]`, _CYratioCyt_ gonna be a key

* Relational Database in Python
	* Normal way
		* Step 1: `engine = create_engine('sqlite:///Chinook.sqlite')`
		* Step 2: `con = engine.connect()`
		* Step 3: `rs = con.execute('SELECT * FROM Album')`
		* Step 4: `df = pd.DataFrame(rs.fetchall())`
		* Step 5: `con.close()`
	* Context manager
```python
engine = create_engine('sqlite:///Chinook.sqlite')
with engine.connect() as con:
	rs = con.execute('SELECT * FROM Album')
	df = pd.DataFrame(rs.fetchall())
	con.close()
```
* 
	* One line connection
```python
engine = create_engine('sqlite:///Chinook.sqlite')
df = pd.read_sql_query('SELECT * FROM Album', engine)
```	
## Introduction and flat files
### Exploring your working directory
* `!ls` in linux-based environment
* `%ls` in Windows

### Importing entire text files
```python
# Open a file: file
file = open('moby_dick.txt', mode='r')

# Print it
print(file.read())

# Check whether file is closed
print(file.closed)

# Close file
file.close()

# Check whether file is closed
print(file.closed)
```
> CHAPTER 1. Loomings.  
>   
> Call me Ishmael. Some years ago--never mind how long precisely--having  
> little or no money in my purse, and nothing particular to interest me on  
> shore, I thought I would sail about a little and see the watery part of  
> the world. It is a way I have of driving off the spleen and regulating  
> the circulation.  
> False  
> True

### Importing text files line by line
* **context manager**: you can bind a variable `file` by using a context manager construct: `with open('huck_finn.txt') as file:`
* While still within this construct, the variable file will be bound to `open('huck_finn.txt')`
```python
# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())
```
> CHAPTER 1. Loomings.  
>   
>   
>   
> Call me Ishmael. Some years ago--never mind how long precisely--having

### Pop quiz: examples of flat files
> which of these file types below is NOT an example of a flat file?  
> A relational database (e.g. PostgreSQL).
### Why we like flat files and the Zen of Python
* [PEP8](https://www.python.org/dev/peps/pep-0008/): is a standard style guide for Python
* It is the basis for how we here to ask our instructors to [style their code](https://www.datacamp.com/teach/documentation#tab_style_guide_python)
* [PEP20](https://www.python.org/dev/peps/pep-0020/), commonly called the Zen of Python
* Acronym `BDFL` stands for, I suggest that you look [here](https://docs.python.org/3.3/glossary.html#term-bdfl).

### Using NumPy to import flat files
* More info about **MNIST** [here](http://yann.lecun.com/exdb/mnist/)
```python
# Import package
import numpy as np

# Assign filename to variable: file
file = 'digits.csv'

# Load file as array: digits
digits = np.loadtxt(file, delimiter=',')

# Print datatype of digits
print(type(digits))

# Select and reshape a row
im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))

# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()
```
![alttext](https://jxjo8w.ch.files.1drv.com/y4muIGzLFR5SoCtK7fk2EUBzWBLF3ua6ZWpuIvVxZgQ78DdBtGErk0-R8ik2YVOLK5Ok4xP5yMwJeFPY4Nbi6BxQi-4oPdl_gO5tf2SzkoeOCTfACwMCYIvSsGX0Q1ZkEIxE_kvNnR-ZnmTcW2xphp2EMQhUSOfgW9XrCcEe-7XCEviMgqAXex8RYSHLYFT2c7bUlKOS8bOTKMlvOQ2p9k4dg/Capture23.png?psid=1)
> <class 'numpy.ndarray'>

### Customizing your NumPy import
* use `','` and `'\t'` for comma-delimited and tab-delimited respectively
* `skiprows` allows you to specify how many rows (not indices) you wish to skip
* `usecols` takes a list of the indices of the columns you wish to keep.
```python
# Import numpy
import numpy as np

# Assign the filename: file
file = 'digits_header.txt'

# Load the data: data
data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0, 2])

# Print data
print(data)
```
> [[ 1.  0.]  
> [ 0.  0.]  
> [ 1.  0.]  
> [ 4.  0.]  
> [ 0.  0.]  
> [ 0.  0.]  
> [ 7.  0.]  
> [ 3.  0.]  
> [ 5.  0.]  
> [ 3.  0.]  
> [ 8.  0.]]

### Importing different datatypes
* These data consists of percentage of sea slug larvae that had metamorphosed in a given time period. Read more [here](http://www.stat.ucla.edu/projects/datasets/seaslug-explanation.html).
* if we tried to import it as using `np.loadtxt()`, Python would throw you a `ValueError: could not convert string to float`. 2 solution for it:
	* set the data type argument `dtype` equal to `str` (for string).
	* skip the first row as we have seen before, using the `skiprows` argument.
```python
# Assign filename: file
file = 'seaslug.txt'

# Import file: data
data = np.loadtxt(file, delimiter='\t', dtype=str)

# Print the first element of data
print(data[0])

# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)

# Print the 10th element of data_float
print(data_float[9])

# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()
```
> ['Time' 'Percent']  
> ['0' '0.357']

![alt text](https://jxjhra.ch.files.1drv.com/y4mvS8N9DdZNNjUpF0d4L99JWd16AzBfldi-jdMiS9F3LYJoyJncwy2DEjyjeCO8PBPVjpGj1JUVPWsonD26cL58vY1D-57yNJtXaeo28H34DmXUFiIb_iwDTwfROiWQBRHaPuRJ3Li1Zh8Tev_dv1K3REVSTtqLxeCZTD1VfgHeKEFJTZWeSR8WIq03ZvfRawKVoZ23eSr-9uwhdZ_V7iq0g/Capture24.png?psid=1)

### Working with mixed datatypes (1)
* Most of the time, we have to import dataset with diferent datatypes and `np.loadxt()` will freak at it
* To solve it, we need to use `np.genfromtxt()` as follows:
`data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)`
	* Third argument tell us there is a header
	* Because the data are of different types, `data` is an object called a [structured array](http://docs.scipy.org/doc/numpy/user/basics.rec.html)
	* Because numpy arrays have to contain elements that are all the same type, the structured array solves this by being a 1D array, where each element of the array is a row of the flat file imported
	> [(890, 1, 1, b'male',  26.  , 0, 0, b'111369',   30.    , b'C148', b'C'),
       (891, 0, 3, b'male',  32.  , 0, 0, b'370376',    7.75  , b'', b'Q')]

### Working with mixed datatypes (2)
* There is also another function `np.recfromcsv()` that behaves similarly to `np.genfromtxt()`, except that its default `dtype` is `None`
```python
# Assign the filename: file
file = 'titanic.csv'

# Import file using np.recfromcsv: d
d= np.recfromcsv(file, delimiter=',', names=True, dtype=None)

# Print out first three entries of d
print(d[:3])
```
> [(1, 0, 3, b'male',  22., 1, 0, b'A/5 21171',   7.25  , b'', b'S')  
> (2, 1, 1, b'female',  38., 1, 0, b'PC 17599',  71.2833, b'C85', b'C')  
> (3, 1, 3, b'female',  26., 0, 0, b'STON/O2. 3101282',   7.925 , b'', b'S')]

### Using pandas to import flat files as DataFrames (1)
* Now we can import dataset with `numpy`, but `DataFrame` in `pandas` is a more appropriate structure in which to store dataset
* DataFrames using the pandas functions `read_csv()` and `read_table()`
```python
# Import pandas as pd
import pandas as pd

# Assign the filename: file
file = 'titanic.csv'

# Read the file into a DataFrame: df
df = pd.read_csv(file)

# View the head of the DataFrame
print(df.head())
```
||Ticket|Fare|Cabin|Embarked|
|--|-------|-------|----|--|
|0|A/5 21171|7.2500|NaN|S|
|1|PC 17599|71.2833|C85|C|
|2|STON/O2. 3101282|7.9250|NaN|S|
|3|113803|53.1000|C123|S|
|4|373450|8.0500|NaN|S|

### Using pandas to import flat files as DataFrames (2)
*  As a bonus, it is then straightforward to retrieve the corresponding `numpy` array using the attribute `values`
```python
# Assign the filename: file
file = 'digits.csv'

# Read the first 5 rows of the file into a DataFrame: data
data = pd.read_csv(file, nrows=5, header=None)

# Build a numpy array from the DataFrame: data_array
data_array = data.values

# Print the datatype of data_array to the shell
print(type(data_array))
```
> [[1 0 3 'male' 22.0 1 0 'A/5 21171' 7.25 nan 'S']
> [2 1 1 'female' 38.0 1 0 'PC 17599' 71.2833 'C85' 'C']
> [3 1 3 'female' 26.0 0 0 'STON/O2. 3101282' 7.925 nan 'S']]

### Customizing your pandas import
* Note that missing values are also commonly referred to as `NA` or `NaN`
*  import a slightly corrupted copy of the Titanic dataset `titanic_corrupt.txt`, which
	* contains comments after the character `#`
	* is tab-delimited.
* `sep`: `pandas` version of `delimiter`
```python
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Assign filename: file
file = 'titanic_corrupt.txt'

# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values=['NA', 'NaN', 'Nothing'])

# Print the head of the DataFrame
print(data.head())

# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()
```
|PassengerId|Survived|Pclass|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
|---|---|---|-----|---|---|---|--------|-------|---|---|
|0|1|0|3|male|22.0|1|0|A/5 21171|7.250|NaN|S|
|1|2|1|1|female|38.0|1|0|PC 17599|NaN|NaN|NaN|

![alt text](https://jxhgoa.ch.files.1drv.com/y4m_csGmCcgyK-pUMXjTnwlnLcLqwFa4avmiI3O-dHCoTvo1PyOTLcfnCdu1Pc-dAXBqRgcBnsj-k2mkA-DE_mhhBbBgVaak3nLctaNet6HOuLwyxtw6pFwjUqOJUlWy8rFNJjGdPA6vlOOoEApyFkoMrRWiYfcMMZKXmVme1s9cMIS8Uzb4yrBkLGzS8M5fBVs_2P2OerHVXAH1nYjTeaXWA/Capture25.png?psid=1)

## Importing data from other file types
### Not so flat any more
* Aside from `!ls` which explores our current directory, [library](https://docs.python.org/2/library/os.html) `os`, which consists of miscellaneous operating system interfaces.
```python
import os
wd = os.getcwd()
os.listdir(wd)
```
> ['titanic_corrupt.txt', 'figure.svg', 'titanic.txt', 'battledeath.xlsx']

### Loading a pickled file
* if you merely want to be able to import them into Python, you can [serialize](https://en.wikipedia.org/wiki/Serialization) them or bytestream
* `rb`: read binary
```python
# Import pickle package
import pickle

# Open pickle file and load data: d
with open('data.pkl', 'rb') as file:
    d = pickle.load(file)

# Print d
print(d)

# Print datatype of d
print(type(d))

```
> {'Aug': '85', 'Mar': '84.4', 'Airline': '8', 'June': '69.4'}  
> <class 'dict'>

### Listing sheets in Excel files
* Modified from the Peace Research Institute Oslo's (PRIO) [dataset](https://www.prio.org/Data/Armed-Conflict/Battle-Deaths/The-Battle-Deaths-Dataset-version-30/)
```python
# Import pandas
import pandas as pd

# Assign spreadsheet filename: file
file = 'battledeath.xlsx'

# Load spreadsheet: xl
xl = pd.ExcelFile(file)

# Print sheet names
print(xl.sheet_names)
```
> ['2002', '2004']

### Importing sheets from Excel files
```python
# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('2004')

# Print the head of the DataFrame df1
print(df1.head())

# Load a sheet into a DataFrame by index: df2
df2 = xl.parse(0)

# Print the head of the DataFrame df2
print(df2.head())
```
||War(country)|2004|
|---|-----------|------------|
|0|Afghanistan|9.451028|
|1|Albania|9.451028|
|2|Algeria|3.407277|
|3|Andorra|0.000000|

### Customizing your spreadsheet import
* Skip the first row of data and name the columns `'Country'` and `'AAM due to War (2002)'` using the argument `names`. The values passed to `skiprows` and `names` all need to be of type list.
* Parse only the first column with the `parse_cols` parameter which just works on **Python 2**, skip the first row and rename the column `'Country'`
```python
# Parse the first sheet and rename the columns: df1
df1 = xl.parse(0, skiprows=1, names=['Country', 'AAM due to War (2002)'])

# Print the head of the DataFrame df1
print(df1.head())

# Parse the first column of the second sheet and rename the column: df2
df2 = xl.parse(1, parse_cols=0, skiprows=1, names=['Country'])

# Print the head of the DataFrame df2
print(df2.head())
```
||Country|AAM due to War (2002)|
|--|---------|-----------|
|0|Albania|0.130354|
|1|Algeria|3.407277|
|2|Andorra|0.000000|
|3|Angola|2.597931|

||Country|
|--|--------|
|0|Albania|
|1|Algeria|
|2|Andorra|
|3|Angola|
### How to import SAS7BDAT
`from sas7bdat import SAS7BDAT`
### Importing SAS files
* The data are adapted from the website of the undergraduate text book [Principles of Economics](http://www.principlesofeconometrics.com/sas/)
```python
# Import sas7bdat package
from sas7bdat import SAS7BDAT

# Save file to a DataFrame: df_sas
with SAS7BDAT('sales.sas7bdat') as file:
    df_sas = file.to_data_frame()

# Print head of DataFrame
print(df_sas.head())

# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[['P']])
plt.ylabel('count')
plt.show()
```
||YEAR|P|S|
|--|-----|-----|------------|
|0|1850.0|12.9|181.899994|
|1|1951.0|11.9|245.000000|
|2|1952.0|10.7|250.199997|
|3|1953.0|11.3|250.199997|
 
![alt text](https://jxgxlg.ch.files.1drv.com/y4m-LiIH3hUebn0vltXbk_pgRNxh22Hj6ZRQn3n2KhG9r-hj7H_Sh965lTtcZvshTRoGxc5nJglfK2bbb1NVGOnkE__Mhi1GT94YfZRZg1jlY5X_7H5Riu_1PAmRkWTsZVl9g3CHtFfnjTm6tt9q2kqJtu5pcT8G0T2B10EgRwnb_UcPUhexqFT87t2ZCAFJTtKaqtAqhaMBE8PHc9Wu5cSvw/Capture26.png?psid=1)
### Using read_stata to import Stata files
* The data consist of disease extents for several diseases in various countries [here](http://www.cid.harvard.edu/ciddata/geog/readme_disarea.html)
`df = pd.read_stata('disarea.dta')`
### Importing Stata files
```python
# Import pandas
import pandas as pd

# Load Stata file into a pandas DataFrame: df
df = pd.read_stata('disarea.dta')

# Print the head of the DataFrame df
print(df.head())

# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of disease')
plt.ylabel('Number of coutries')
plt.show()
```
||wbcode|country|disa1|disa2|...|disa25|
|--|----|----------|----|----|---|----|
|0|AFG|Afghanistan|0.00|0.00|....|0.00|
|1|AGO|Angola|0.32|0.02|....|0.00|

![alt text](https://jxh8iq.ch.files.1drv.com/y4mnlJdbvp7-TcXn-ASpDNkVv2jTexn8ld_6xLazEXSIyh3F6eRc022t63tpW2Ky9PzJVQRtNyX_iLPEaOCCz-mEMShSDz4qG-8Ja5pVUEhbuFjksWlciIbHzuDOpnReiD5DGb6pmQYOIgVy4nNCS11sMJ-ToWgeCdTovmkMdO9hHe5aCPIhq0nRvmxkOwIG7TjS3mlQVetb0hPbO4rJJu01g/Capture27.png?psid=1)
### Structure of HDF5 files (Hierarchical Data File ver 5)
![alt text](https://jxh12g.ch.files.1drv.com/y4mmi4WeCOWHjZyp46_BFq3ooHO-KwLCZATTJqrr_c1vBLprS_KeSSrzjvU6S1OkGpks_2pKbR0N8aF-I1OGjvhEAsobW2_-I_5hF4Fk4nKIcgyFaIlYcvHFLHCKlcHDzF-piqVyFbuXhkSG-hq9QkX3AZi_n6_2YFXEDVpVQH_9O2Yv4Q8ycIc1fWbbk6lDBTMOPPdLWjGTk1ib6y3O_l3Ug/Capture28.PNG?psid=1)
### Using File to import HDF5 files
`h5py_data = h5py.File(h5py_file, 'r')`
### Using h5py to import HDF5 files
* LIGO data plus loads of documentation and tutorials [here](https://losc.ligo.org/events/GW150914/). There is also a great tutorial on Signal Processing with the data [here](https://losc.ligo.org/s/events/GW150914/GW150914_tutorial.html).
```python
# Import packages
import numpy as np
import h5py

# Assign filename: file
file = 'LIGO_data.hdf5'

# Load file: data
data = h5py.File(file, 'r')

# Print the datatype of the loaded file
print(type(data))

# Print the keys of the file
for key in data.keys():
    print(key)
```
> <class 'h5py._hl.files.File'>
	
> meta  
> quality  
> strain

### Extracting data from your HDF5 file
![alt text](https://9txe2w.ch.files.1drv.com/y4mc5m2V4Lkf0Eh6r0ampuI8FB08A36OAj87F_-k_7uglY7qz82kXgPD4m7x-gRo7AJ3b7_dPjeSHRj8z_6QNt1UbRa3wRW9xEVVyupXG3sERCiXnyk4WlyqtBOMr3QkIZF8kTwzxH2rqzCBWw9ZbwB5XXsKKkF6d0rgAeLM_RasXKWhsAw_wHpXmQaCfrSIM2sz7FXr1Ed3X90vqjrkyRsDA/Capture31.png?psid=1)
```python
# Get the HDF5 group: group
group = data['strain']

# Check out keys of group
for key in group.keys():
    print(key)

# Set variable equal to time series data: strain
strain = data['strain']['Strain'].value

# Set number of time points to sample: num_samples
num_samples = 10000

# Set time vector
time = np.arange(0, 1, 1/num_samples)

# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()
```
> Strain

![alt text](https://jxjazq.ch.files.1drv.com/y4mEuSeTpoHQfKsaDbWiCPfEemYZvzHu85qsfpAXulMflXykAo3l7jgQH34Zf4BMz4jtluwaBfKC4usiuDQ87zXQ9mutG2b-ntrPpDl5tHfXVeG_fR27VSSOIMbNPsu1XIoBp_HNNyElV4ccM5ZLkukA_0mBYMbHDKE69SGC6Pwi7fED7OSgD3kktVCeNN5wGLFQLrO_KEh6MTxdem5e6Oy5g/Capture29.png?psid=1)

### Loading .mat files
* File contains [gene expression data](https://www.mcb.ucdavis.edu/faculty-labs/albeck/workshop.htm) from the Albeck Lab at UC Davis. We can find the data and some great documentation [here](https://www.mcb.ucdavis.edu/faculty-labs/albeck/workshop.htm).
```python
# Import package
import scipy.io

# Load MATLAB file: mat
mat = scipy.io.loadmat('albeck_gene_expression.mat')

# Print the datatype type of mat
print(type(mat))
```
> <class 'dict'>  

### The structure of `.mat` in Python
```python
# Print the keys of the MATLAB dictionary
print(mat.keys())

# Print the type of the value corresponding to the key 'CYratioCyt'
print(type(mat['CYratioCyt']))

# Print the shape of the value corresponding to the key 'CYratioCyt'
print(np.shape(mat['CYratioCyt']))

# Subset the array and plot it
data = mat['CYratioCyt'][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel('time (min.)')
plt.ylabel('normalized fluorescence (measure of expression)')
plt.show()
```
> dict_keys(['__header__', 'yfpNuc', 'yfpCyt', 'cfpCyt', 'cfpNuc', '__version__', 'rfpNuc', '__globals__', 'CYratioCyt', 'rfpCyt'])  
> <class 'numpy.ndarray'>  
> (200, 137)

![alt text](https://9tvdzw.ch.files.1drv.com/y4mxXV-ZFccDwuvd_WKP1hNbXorThktPlgraqvnJ6LvM6jE9F1lZT1z7Ko0SmjxRiygImpL8I5v7j8qZVa_TmdNRxdPKDssg-UZc7pjozdHcA9i_kNPaemaWEvcHRTHRIzE7y8F-cuBVp96CQ8ORpAtlkoMmuwk1wn4-9Ld5VglK2kG1HFKutkDFzZ6RoG7j5LfDclgdnXPFz7KvDp6B3O6jA/Capture30.png?psid=1)

## Working with relational databases in Python
### Creating a database engine
* A little bit of background on the [Chinook database](http://chinookdatabase.codeplex.com/): the Chinook database contains information about a semi-fictional digital media store in which media data is real and customer, employee and sales data has been manually created.
* `'sqlite:///Northwind.sqlite'` is called the _connection string_ to the SQLite database Northwind.sqlite
```python
# Import necessary module
from sqlalchemy import create_engine

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
```

### What are the tables in the database?
```python
# Import necessary module
from sqlalchemy import create_engine

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Save the table names to a list: table_names
table_names = engine.table_names()

# Print the table names to the shell
print(table_names)
```
> ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']

### The Hello World of SQL Queries!
```python
# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Open engine connection: con
con = engine.connect()

# Perform query: rs
rs = con.execute('SELECT * FROM Album')

# Save results of the query to DataFrame: df
df = pd.DataFrame(rs.fetchall())

# Close connection
con.close()

# Print head of DataFrame df
print(df.head())
```
### Customizing the Hello World of SQL Queries
* `rs.fetchmany(size=3)` only fetch 3 rows
```python
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute('SELECT LastName, Title from Employee')
    df = pd.DataFrame(rs.fetchmany(size=3))
    df.columns = rs.keys()

# Print the length of the DataFrame df
print(len(df))

# Print the head of the DataFrame df
print(df.head())
```
> 3

||LastName|Title|
|--|-----|-----------------|
|0|Adams|General Manager|
|1|Edwards|Sales Manager|
|2|Peacock|Sales Support Agent|

### Filtering your database records using SQL's WHERE
```python
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute('SELECT * FROM Employee WHERE EmployeeId >= 6')
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print the head of the DataFrame df
print(df.head())
```

||EmployeeId|LastName|FirstName|Title|ReportsTo|BirthDate|
|--|--|--------|---------|------|--|------------------|
|0|6|Mitchell|Michael|IT Manager|1|1973-07-01 00:00:00|
### Ordering your SQL records with ORDER BY
```python
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Open engine in context manager
with engine.connect() as con:
    rs = con.execute('SELECT * FROM Employee ORDER BY BirthDate')
    df = pd.DataFrame(rs.fetchall())

    # Set the DataFrame's column names
    df.columns = rs.keys()

# Print head of DataFrame
print(df.head())
```
||EmployeeId|LastName|FirstName|Title|ReportsTo|BirthDate|
|--|--|--------|---------|------|--|------------------|
|0|6|Mitchell|Michael|IT Manager|1|1973-07-01 00:00:00|
### Pandas and The Hello World of SQL Queries!
* We can replace 4 lines: create engine, execute query based on this engine, fetch all to DataFrame and assign columns of this DataFrame by: `df = pd.read_sql_query("SELECT * FROM Orders", engine)
`
```python
# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Execute query and store records in DataFrame: df
df = pd.read_sql_query('SELECT * FROM Album', engine)

# Print head of DataFrame
print(df.head())

# Open engine in context manager
# Perform query and save results to DataFrame: df1
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()

# Confirm that both methods yield the same result: does df = df1 ?
print(df.equals(df1))
```

||AlbumId|Title|ArtistId|
|--|--|--------------------------------|--|
|0|1|For Those About To Rock We Salute You|1|
|1|2|Balls to the Wall|2|

### Pandas for more complex querying
```python
# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Execute query and store records in DataFrame: df
df = pd.read_sql_query('SELECT * FROM Employee WHERE EmployeeId >= 6 ORDER BY BirthDate', engine)

# Print head of DataFrame
print(df.head())
```
### The power of SQL lies in relationships between tables: INNER JOIN
```python
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute('SELECT Title, Name FROM Album INNER JOIN Artist ON Album.ArtistID = Artist.ArtistID')
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print head of DataFrame df
print(df.head())
```
### Filtering your INNER JOIN
```python
# Execute query and store records in DataFrame: df
# Execute query and store records in DataFrame: df
df = pd.read_sql_query('SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000', engine)

# Print head of DataFrame
print(df.head())
```