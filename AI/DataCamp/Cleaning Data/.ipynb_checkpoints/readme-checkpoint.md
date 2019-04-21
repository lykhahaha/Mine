# Cleaning Data
## Exploring your data
### Loading and viewing your data
* you're going to look at a subset of the Department of Buildings Job Application Filings dataset from the [NYC Open Data](http://opendata.cityofnewyork.us/) portal. This dataset consists of job applications filed on January 22, 2017.
```python
# Import pandas
import pandas as pd

# Read the file into a DataFrame: df
df = pd.read_csv('dob_job_application_filings_subset.csv')

# Print the head of df
print(df.head())

# Print the tail of df
print(df.tail())

# Print the shape of df
print(df.shape)

# Print the columns of df
print(df.columns)

# Print the head and tail of df_subset
print(df_subset.head())
print(df_subset.tail())
```

### Further diagnosis
* The `.info()` method provides important information about a DataFrame, such as the number of rows, number of columns, number of non-missing values in each column, and the data type stored in each column
```python
# Print the info of df
print(df.info())

# Print the info of df_subset
print(df_subset.info())
```
> RangeIndex: 12846 entries, 0 to 12845  
> Data columns (total 82 columns):  
> Job #                           12846 non-null int64  
> Doc #                           12846 non-null int64  
> Borough                         12846 non-null object  
> House #                         12846 non-null object

### Frequency counts for categorical data
* `.describe()` can only be used on numeric columns
* So how can you diagnose data issues when you have categorical data? 
* One way is by using the `.value_counts()` method, which returns the frequency counts for each unique value in a column!
* This method also has an optional parameter called dropna which is `True` by default.
* You want to set the `dropna` column to `False` so if there are missing values in a column, it will give you the frequency counts.
```python
# Print the value counts for 'Borough'
print(df['Borough'].value_counts(dropna=False))

# Print the value_counts for 'State'
print(df['State'].value_counts(dropna=False))

# Print the value counts for 'Site Fill'
print(df['Site Fill'].value_counts(dropna=False))
```
> MANHATTAN        6310  
> BROOKLYN         2866  
> QUEENS           2121  
> BRONX             974  
> STATEN ISLAND     575  
> Name: Borough, dtype: int64  
> NY    12391  
> NJ      241  
> PA       38  
> CA       20  
> OH       19
> IL       17  
> FL       17  
> CT       16  
> Name: State, dtype: int64  
> NOT APPLICABLE                              7806  
> NaN                                         4205  
> ON-SITE                                      519  
> OFF-SITE                                     186  
> USE UNDER 300 CU.YD                          130  
> Name: Site Fill, dtype: int64

### Visualizing single variables with histograms
* Visualizing single variables with histograms
* You'll notice that there are extremely large differences between the `min` and `max` values, and the plot will need to be adjusted accordingly. 
* In such cases, it's good to look at the plot on a log scale. The keyword arguments `logx=True` or `logy=True` can be passed in to `.plot()`
```python
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Plot the histogram
df['Existing Zoning Sqft'].plot(kind='hist', rot=70, logx=True, logy=True)

# Display the histogram
plt.show()
```
![alt text](https://hjexug.ch.files.1drv.com/y4m_AKvKCbDz-GV1OLk8E8-4ho0PLCD12RexlLNSC-KQ2JE9RihbrJSWs47CBn8bCHouOyd3_xCJeolpq3cQLIE6dnI8y1j8HvX-g6In_4lKx6VIKzyvYHMyP5pO18DC2CgSOJpJ_AMhS69fWUcbNjWXKIZ5IPjMhkJr6SxIo2eWmCEvxxG1oBOv-uWZ3Xvq163qXNHL2Ur2f5M99Jlx1v1HQ/Capture34.png?psid=1)

### Visualizing multiple variables with boxplots
* To visualize multiple variables, boxplots are useful, especially when one of the variables is categorical.
```python
# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt

# Create the boxplot
df.boxplot(column='initial_cost', by='Borough', rot=90)

# Display the plot
plt.show()
```
![alt text](https://hjf83q.ch.files.1drv.com/y4mEFX4gEHVc1Q4nbRERuov8joF2-4pjb_BO944-Vf7ugJJ1_gvT_jB-Ja9B4pURj8uXkWA2e3T-fDcIZKJTnrN7ykKaw_B8GmJxUcfF1PoBZPAsLXCcM0kklFeuz9YWyycPVIYRRW2PbyNEMVuEAkZ-0AQ1e_HV-Hry8CM9T2W3x3e6nFYhuiMp2RFW-dyKHTb8q_kbYOlMqqzOir61U-Bfg/Capture35.png?psid=1)

### Visualizing multiple variables with scatter plots
* When you want to visualize two numeric columns, scatter plots are ideal.
* Since these outliers dominate the plot, an additional DataFrame, `df_subset`, has been provided, in which some of the extreme values have been removed. 
* After making a scatter plot using this, you'll find some interesting patterns here that would not have been seen by looking at summary statistics or 1 variable plots.
```python
# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt

# Create and display the first scatter plot
df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()

# Create and display the second scatter plot
df_subset.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()
```

![alt text](https://hjgyxg.ch.files.1drv.com/y4mEAH3zhORqxfht9ta6ZYxGTXgQyEyDucZtoSgx0bqk8LJbneyafyCKKQwsBkxf7ANYOslCoSwwO9zs6kBcIovRVHbeWxJu4vNYs32B3Wfs7z1JRDJkMfBELU_V8rRXnhFVc9zuXgvplYFclgLOz7HRNzSL9VsKMmrnjQ5MIETE51P9oLL6JkREKdXUKbeeGWm9QH2JJNkMQkRwbva5H5t8A/Capture36.png?psid=1)

![alt text](https://hjfnow.ch.files.1drv.com/y4mz0d6HoxeZKuK-KYpr8Ad5Ffa5FXcFWgSmptq2EVLCLEGnxlvVMG_wYtLmSiMVCdPsqMYrzpW6wEIAeNsOvXqV-4gY2hnSN3zJRvIjsTftM2sAKQ4AH-2zOjedBT51Kht1SFw6KSJHVQeYXCfccZYJIMVNZ7W4Ljlt7qNSvVucrzUXiPqvGRyun3tfm57FAg076dwJMkQ43VmQASnIoo2Jg/Capture37.png?psid=1)

## Tidying data for analysis
### Recognizing tidy data
* For data to be tidy, it must have:
	* Each variable as a separate column.
	* Each row as a separate observation.

### Reshaping your data using melt
![alt text](https://hjeqrq.ch.files.1drv.com/y4muJBLynmwijk2Ph7BJ-OodnlkY6WJRF1z0yZP66YhAS7Guj1IchDsgMFOmZPWWpoRSobDKyOD9TGQY-zoCm6opjoIsiugYRkpc2w72EcFJZsyFCgJAmdCgngHKibLbrqo_NeR1X7UjoTInOc6WUXlq6rw7yJqoZv6XBVAFPQrG-QIKm0JvnEk9XwcrfZ5dabm9cGHMGu_UOLOev-4QbLPEg/Capture39.png?psid=1)
* Melting data is the process of turning columns of your data into rows of data
* In the tidy DataFrame, the variables `Ozone`, `Solar.R`, `Wind`, and `Temp` each had their own column. If, however, you wanted these variables to be in rows instead, you could melt the DataFrame
* But This is important to keep in mind: Depending on how your data is represented, you will have to reshape it differently.
* Two parameters you should be aware of: `id_vars` and `value_vars`
* The `id_vars` represent the columns of the data you do not want to melt (i.e., keep it in its current shape)
* The `value_vars` represent the columns you **do** wish to melt into rows
* By default, if **no** `value_vars` are provided, all columns not set in the `id_vars` will be melted
```python
# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'])

# Print the head of airquality_melt
print(airquality_melt.head())
```
|Ozone|Solar.R|Wind|Temp|Month|Day|
|---|------|-----|----|----|----|
|41.0|190.0|7.4|67|5|1|
|36.0|118.0|8.0|72|5|2|
|12.0|149.0|12.6|74|5|3|
|18.0|313.0|11.5|62|5|4|
|NaN|NaN|14.3|56|5|5|

|Month|Day|variable|value|
|---|----|-------|-----|
|5|1|Ozone|41.0|
|5|2|Ozone|36.0|
|5|3|Ozone|12.0|
|5|4|Ozone|18.0|
|5|5|Ozone|NaN|

### Customizing melted data
* Melt the `Ozone`, `Solar.R`, `Wind`, and `Temp` columns of `airquality` into rows, with the default `variable` column renamed to `'measurement'` and the default value column renamed to `'reading'`.
```python
# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')

# Print the head of airquality_melt
print(airquality_melt.head())
```
|Ozone|Solar.R|Wind|Temp|Month|Day|
|---|------|-----|----|----|----|
|41.0|190.0|7.4|67|5|1|
|36.0|118.0|8.0|72|5|2|
|12.0|149.0|12.6|74|5|3|
|18.0|313.0|11.5|62|5|4|
|NaN|NaN|14.3|56|5|5|

|Month|Day|measurement|reading|
|----|------|--------|------|
|5|1|Ozone|41.0|
|5|2|Ozone|36.0|
|5|3|Ozone|12.0|
|5|4|Ozone|18.0|
|5|5|Ozone|NaN|

### Pivot data
* Pivoting data is the opposite of melting it
* `index` parameter specify the columns that you _don't_ want pivoted which is similar to the `id_vars` parameter of `pd.melt()`
* `columns` (the name of the column you want to pivot), and `values` (the values to be used when the column is pivoted)
```python
# Print the head of airquality_melt
print(airquality_melt.head())

# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')

# Print the head of airquality_pivot
print(airquality_pivot.head())
```
|Month|Day|measurement|reading|
|----|------|--------|------|
|5|1|Ozone|41.0|
|5|2|Ozone|36.0|
|5|3|Ozone|12.0|
|5|4|Ozone|18.0|
|5|5|Ozone|NaN|

|measurement|Ozone|Solar.R|Temp|Wind|
|----------|------|------|-------|------|
|Month&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Day|||||                              
|9&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;26|30.0|193.0|70.0|6.9|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;27|NaN|145.0|77.0|13.2|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;28|14.0|191.0|75.0|14.3|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;29|18.0|131.0|76.0|8.0|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;30|20.0|223.0|68.0|11.5|

### Resetting the index of a DataFrame
* After pivoting `airquality_melt` in the previous exercise, you didn't quite get back the original DataFrame.
* What you got back instead was a pandas DataFrame with a [hierarchical index (also known as a MultiIndex)](http://pandas.pydata.org/pandas-docs/stable/advanced.html).  In essence, they allow you to group columns or rows by another variable - in this case, by `'Month'` as well as `'Day'`.
```python
# Print the index of airquality_pivot
print(airquality_pivot.index)

# Reset the index of airquality_pivot: airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the new index of airquality_pivot
print(airquality_pivot.index)

# Print the head of airquality_pivot
print(airquality_pivot.head())
```
> RangeIndex(start=0, stop=153, step=1)

|measurement|Month|Day|Ozone|Solar.R|Temp|Wind|
|---|----|-----|-----|-----|----|-----|
|0|5|1|41.0|190.0|67.0|7.4|
|1|5|2|36.0|118.0|72.0|8.0|
|2|5|3|12.0|149.0|74.0|12.6|
|3|5|4|18.0|313.0|62.0|11.5|
|4|5|5|NaN|NaN|56.0|14.|

### Pivoting duplicate values
```python
# Pivot airquality_dup: airquality_pivot
airquality_pivot = airquality_dup.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading', aggfunc=np.mean)

# Reset the index of airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the head of airquality_pivot
print(airquality_pivot.head())

# Print the head of airquality
print(airquality.head())
```
|measurement|Month|Day|Ozone|Solar.R|Temp|Wind|
|----|----|----|----|------|-----|-------|
|0|5|1|41.0|190.0|67.0|7.4|
|1|5|2|36.0|118.0|72.0|8.0|
|2|5|3|12.0|149.0|74.0|12.6|
|3|5|4|18.0|313.0|62.0|11.5|
|4|5|5|NaN|NaN|56.0|14.3|

|Ozone|Solar.R|Wind|Temp|Month|Day|
|---|-----|----|----|-----|----|
|0|41.0|190.0|7.4|67|5|1|
|1|36.0|118.0|8.0|72|5|2|
|2|12.0|149.0|12.6|74|5|3|
|3|18.0|313.0|11.5|62|5|4|
|4|NaN|NaN|14.3|56|5|5|

### Splitting a column with .str
* The dataset you saw in the video, consisting of case counts of tuberculosis by country, year, gender, and age group, has been pre-loaded into a DataFrame as `tb`.
* Tidy the `'m014'` column, which represents males aged 0-14 years of age
* Need to extract the first letter into a new column for `gender`, and the rest into a column for `age_group`
* Slicing by using the `str` attribute of columns of type `object`.
```python
type(df['Initial Cost'].str)
```
> <class 'pandas.core.strings.StringMethods'>

```python
# Melt tb: tb_melt
print(tb.head())
tb_melt = pd.melt(tb, id_vars=['country', 'year'])

# Create the 'gender' column
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Print the head of tb_melt
print(tb_melt.head())
```
![alt text](https://hjgkoa.ch.files.1drv.com/y4mz5ALWlb6mPdMAJ6r6ObMsMrv4jEyM97Vn28f2Oqke-HU1sNkkBtvl4b5U8hVK8KJv2GuY95U26JGrXq0ut9G1ky2w7YvZyHoetFH-VRZRI1dftB_NDcBxX-2Cxh9znnwlbakVQaqnoXCPB6tQTV2-AghWrMbGyxR45WPn_fqAAD3_tfHSZTFdJnvIoaPfmio2SG2yN0mVGWVWnJ5AJvIxw/Capture40.PNG?psid=1)

|country|year|variable|value|gender|age_group|
|---|----|----|-----|-----|-----|
|AD|2000|m014|0.0|m|014|
|AE|2000|m014|2.0|m|014|
|AF|2000|m014|52.0|m|014|
|AG|2000|m014|.0|m|014|
|AL|2000|m014|2.0|m|014|

### Splitting a column with .split() and .get()
*  You'll learn how to deal with such cases in this exercise, [using a dataset consisting of Ebola cases and death counts by state and country](https://data.humdata.org/dataset/ebola-cases-2014). It has been pre-loaded into a DataFrame as `ebola`.
* `ebola.columns`
* `Index(['Date', 'Day', 'Cases_Guinea', 'Cases_Liberia', 'Cases_SierraLeone', 'Cases_Nigeria', 'Cases_Senegal', 'Cases_UnitedStates', 'Cases_Spain', 'Cases_Mali', 'Deaths_Guinea', 'Deaths_Liberia', 'Deaths_SierraLeone', 'Deaths_Nigeria', 'Deaths_Senegal', 'Deaths_UnitedStates', 'Deaths_Spain', 'Deaths_Mali'], dtype='object')`
* Here, the underscore _ serves as a delimiter between the first part (cases or deaths), and the second part (country)
* This time, you cannot directly slice the variable by position as in the previous exercise. You now need to use Python's built-in string method called `.split()`
* You can do this on `Cases_Guinea`, for example, using `Cases_Guinea.split('_')`, which returns the list `['Cases', 'Guinea']`
```python
# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')

# Create the 'str_split' column
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

print(type(ebola_melt.loc[1, 'str_split']))

# Create the 'type' column
ebola_melt['type'] = ebola_melt.str_split.str.get(0)

# Create the 'country' column
ebola_melt['country'] = ebola_melt.str_split.str.get(1)

# Print the head of ebola_melt
print(ebola_melt.head())
```
<class 'list'>

|Date|Day|type_country|counts|str_split|type|country|
|---------|----|------------|-------|---------------|------|------|
|1/5/2015|289|Cases_Guinea|2776.0|[Cases, Guinea]|Cases|Guinea|
|1/4/2015|288|Cases_Guinea|2775.0|[Cases, Guinea]|Cases|Guinea|
|1/3/2015|287|Cases_Guinea|2769.0|[Cases, Guinea]|Cases|Guinea|
|1/2/2015|286|Cases_Guinea|NaN|[Cases, Guinea]|Cases|Guinea|
|12/31/2014|284|Cases_Guinea|2730.0|[Cases, Guinea]|Cases|Guinea|

## Combining data for analysis
### Combining rows of data
* The dataset you'll be working with here relates to [NYC Uber data](http://data.beta.nyc/dataset/uber-trip-data-foiled-apr-sep-2014)
* Three DataFrames have been pre-loaded: 
    * `uber1`, which contains data for April 2014
    * `uber2`, which contains data for May 2014
    * `uber3`, which contains data for June 2014
* This exercise is to concatenate these DataFrames together such that the resulting DataFrame has the data for all three months.
```python
# Concatenate uber1, uber2, and uber3: row_concat
row_concat = pd.concat([uber1, uber2, uber3])

# Print the shape of row_concat
print(row_concat.shape)

# Print the head of row_concat
print(row_concat.head())
```

||Date/Time|Lat|Lon|Base|
|--|---------------------|-----|------|-----|
|0|4/1/2014 0:11:00|40.7690|-73.9549|B02512|
|1|4/1/2014 0:17:00|40.7267|-74.0345|B02512|
|2|4/1/2014 0:21:00|40.7316|-73.9873|B02512|
|3|4/1/2014 0:28:00|40.7588|-73.9776|B02512|
|4|4/1/2014 0:33:00|40.7594|-73.9722|B02512|

### Combining columns of data
* Stitching data together from the sides instead of the top and bottom
* Use the same `pd.concat()` function, but this time with the keyword argument `axis=1` for `attribute`
* The default, `axis=0` for `observation`, is for a row-wise concatenation
```python
# Concatenate ebola_melt and status_country column-wise: ebola_tidy
ebola_tidy = pd.concat([ebola_melt, status_country], axis=1)

# Print the shape of ebola_tidy
print(ebola_tidy.shape)

# Print the head of ebola_tidy
print(ebola_tidy.head())
```

||Date|Day|status_country|counts|status|country|
|-----------|----|----------|----|-----|-------|-------|
|0|1/5/2015|289|Cases_Guinea|2776.0|Cases|Guinea|
|1|1/4/2015|288|Cases_Guinea|2775.0|Cases|Guinea|
|2|1/3/2015|287|Cases_Guinea|2769.0|Cases|Guinea|
|3|1/2/2015|286|Cases_Guinea|NaN|Cases|Guinea|
|4|12/31/2014|284|Cases_Guinea|2730.0|Cases|Guinea|

### Finding files that match a pattern
* What if you have 100 .csv file, and want to concatenate to one file
* `glob` that takes a pattern and returns a list of the files in the working directory that match that pattern.
* Then you use `pd.concat([])` which has the list `glob` reutrns in `[]`
* Now the pattern of 100 .csv file is `part_*.csv`

```python
# Import necessary modules
import glob
import pandas as pd

# Write the pattern: pattern
pattern = '*.csv'

# Save all file matches: csv_files
csv_files = glob.glob(pattern)

# Print the file names
print(csv_files)

# Load the second file into a DataFrame: csv2
csv2 = pd.read_csv(csv_files[2])

# Print the head of csv2
print(csv2.head())
```
> ['uber-raw-data-2014_06.csv', 'uber-raw-data-2014_04.csv', 'uber-raw-data-2014_05.csv']

### Iterating and concatenating all matches
```python
# Create an empty list: frames
frames = []

#  Iterate over csv_files
for csv in csv_files:

    #  Read csv into a DataFrame: df
    df = pd.read_csv(csv)
    
    # Append df to frames
    frames.append(df)

# Concatenate frames into a single DataFrame: uber
uber = pd.concat(frames)

# Print the shape of uber
print(uber.shape)

# Print the head of uber
print(uber.head())
```
||Date/Time|Lat|Lon|Base|
|--|---------------------|-----|------|-----|
|0|4/1/2014 0:11:00|40.7690|-73.9549|B02512|
|1|4/1/2014 0:17:00|40.7267|-74.0345|B02512|
|2|4/1/2014 0:21:00|40.7316|-73.9873|B02512|
|3|4/1/2014 0:28:00|40.7588|-73.9776|B02512|
|4|4/1/2014 0:33:00|40.7594|-73.9722|B02512|

### 1-to-1 data merge
* Merging data allows you to combine disparate datasets into a single dataset to do more complex analysis.
* The dataset was taken from a sqlite database from the [Software Carpentry SQL lesson](http://swcarpentry.github.io/sql-novice-survey/).
> site.head()

||name|lat|long|
|---|-----|-------|-----|
|0|DR-1|-49.85|-128.57|
|1|DR-3|-47.15|-126.72|
|2|MSK-4|-48.87|-123.40|

> visited.head()

||ident|site|dated|
|---|-----|-------|-----|
|0|619|DR-1|1927-02-08|
|1|734|DR-3|1939-01-07|
|2|837|MSK-4|1932-01-14|

* Task is to perform a 1-to-1 merge of these two DataFrames using the `'name'` column of `site` and the `'site'` column of `visited`
```python
# Merge the DataFrames: o2o
o2o = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# Print o2o
print(o2o.head())
```
||name|lat|long|ident|site|dated|
|---|-----|-----|-------|---|---|------------|
|0|DR-1|-49.85|-128.57|619|DR-1|1927-02-08|
|1|DR-3|-47.15|-126.72|734|DR-3|1939-01-07|
|2|MSK-4|-48.87|-123.40|837|MSK-4|1932-01-14|

### Many-to-1 data merge
> site.head()

||name|lat|long|
|---|-----|-------|-----|
|0|DR-1|-49.85|-128.57|
|1|DR-3|-47.15|-126.72|
|2|MSK-4|-48.87|-123.40|

> visited.head()

||ident|site|dated|
|---|-----|-------|-----|
|0|619|DR-1|1927-02-08|
|1|622|DR-1|1927-02-10|
|2|734|DR-3|1939-01-07|
|3|735|DR-3|1930-01-12|
|4|751|DR-3|1930-02-26|

```python
# Merge the DataFrames: m2o
m2o = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# Print m2o
print(m2o.head())
```
||name|lat|long|ident|site|dated|
|---|-----|-----|-------|---|---|------------|
|0|DR-1|-49.85|-128.57|619|DR-1|1927-02-08|
|1|DR-1|-49.85|-128.57|622|DR-1|1927-02-10|
|2|DR-1|-49.85|-128.57|844|DR-1|1932-03-22|
|3|DR-3|-47.15|-126.72|734|DR-3|1939-01-07|
|4|DR-3|-47.15|-126.72|735|DR-3|1930-01-12|

### Many-to-many data merge
> survey.head()

||taken|person|quant|reading|
|--|----|----|---|-----|
|0|619|dyer|rad|9.82|
|1|619|dyer|sal|0.13|
|2|622|dyer|rad|7.80|
|3|622|dyer|sal|0.09|
|4|734|pb|rad|8.41|
```python
# Merge site and visited: m2m
m2m = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# Merge m2m and survey: m2m
m2m = pd.merge(left=m2m, right=survey, left_on='ident', right_on='taken')

# Print the first 20 lines of m2m
print(m2m.head(20))
```

||name|lat|long|ident|site|dated|taken|person|quant|reading|
|----|------|-------|--------|--------|--------|--------|------|--------|-------|-----|
|0|DR-1|-49.85|-128.57|619|DR-1|1927-02-08|619|dyer|rad|9.82|
|1|DR-1|-49.85|-128.57|619|DR-1|1927-02-08|619|dyer|sal|0.13|
|2|DR-1|-49.85|-128.57|622|DR-1|1927-02-10|622|dyer|rad|7.80|
|3|DR-1|-49.85|-128.57|622|DR-1|1927-02-10|622|dyer|sal|0.09|
|4|DR-1|-49.85|-128.57|844|DR-1|1932-03-22|844|roe|rad|11.25|
|5|DR-3|-47.15|-126.72|734|DR-3|1939-01-07|734|pb|rad|8.41|
|6|DR-3|-47.15|-126.72|734|DR-3|1939-01-07|734|lake|sal|0.05|
|7|DR-3|-47.15|-126.72|734|DR-3|1939-01-07|734|pb|temp|-21.50|
|8|DR-3|-47.15|-126.72|735|DR-3|1930-01-12|735|pb|rad|7.22|
|9|DR-3|-47.15|-126.72|735|DR-3|1930-01-12|735|NaN|sal|0.06|
|10|DR-3|-47.15|-126.72|735|DR-3|1930-01-12|735|NaN|temp|-26.00|
|11|DR-3|-47.15|-126.72|751|DR-3|1930-02-26|751|pb|rad|4.35|
|12|DR-3|-47.15|-126.72|751|DR-3|1930-02-26|751|pb|temp|-18.50|
|13|DR-3|-47.15|-126.72|751|DR-3|1930-02-26|751|lake|sal|0.10|
|14|DR-3|-47.15|-126.72|752|DR-3|NaN|752|lake|rad|2.19|
|15|DR-3|-47.15|-126.72|752|DR-3|NaN|752|lake|sal|0.09|
|16|DR-3|-47.15|-126.72|752|DR-3|NaN|752|lake|temp|-16.00|
|17|DR-3|-47.15|-126.72|752|DR-3|NaN|752| roe|sal|41.60|
|18|MSK-4|-48.87|-123.40|837|MSK-4|1932-01-14|837|lake|rad|1.46|
|19|MSK-4|-48.87|-123.40|837|MSK-4|1932-01-14|837|lake|sal|0.21|

## Cleaning data for analysis
### Converting data types
* All `category` variables reduces memory usage.
```python
# Convert the sex column to type 'category'
tips.sex = tips.sex.astype('category')

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype('category')

# Print the info of tips
print(tips.info())
```
> <class 'pandas.core.frame.DataFrame'>  
> RangeIndex: 244 entries, 0 to 243  
> Data columns (total 7 columns):  
> total_bill    244 non-null float64  
> tip           244 non-null float64  
> sex           244 non-null category  
> smoker        244 non-null category  
> day           244 non-null object  
> time          244 non-null object  
> size          244 non-null int64  
> dtyes: category(2), float64(2), int64(1), object(2)  
> memory usage: 10.1+ KB  
> None  

### Working with numeric data
* If you expect type of column is numeric (`int` or `float`), but instead it's `object`
* To clear non-numeric columns, we use `pd.to_numeric()` to `ignore` which ingore missing value or `coerce` which sets missing value as  bad value or use techniques in **Chapter 1**
```python
# Convert 'total_bill' to a numeric dtype
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')

# Convert 'tip' to a numeric dtype
tips['tip'] = pd.to_numeric(tips['tip'], errors='coerce')

# Print the info of tips
print(tips.info())
```
> <class 'pandas.core.frame.DataFrame'>  
> RangeIndex: 244 entries, 0 to 243  
> Data columns (total 7 columns):  
> total_bill    202 non-null float64  
> tip           220 non-null float64  
> sex           234 non-null category  
> smoker        229 non-null category  
> day           243 non-null category  
> time          227 non-null category  
> size          231 non-null float64  
> dtypes: category(4), float64(3)  
> memory usage: 6.9 KB  
> None  

### String parsing with regular expressions
`re.compile` is to help you reuse the pattern for next string
```python
# Import the regular expression module
import re

# Compile the pattern: prog
prog = re.compile('\d{3}-\d{3}-\d{4}')

# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result))

# See if the pattern matches
result = prog.match('1123-456-7890')
print(bool(result))
```
> True  
> False

### Extracting numerical values from strings
* Suppose you have `'the recipe calls for 6 strawberries and 2 bananas'` and want to extract `6` and `2` to compare strawberry and banana
* `\d+`: ensures that `10` is viewed as one number and not as `1` and `0`
```python
# Import the regular expression module
import re

# Find the numeric values: matches
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)
```
> ['10', '1']

### Pattern matching
* Use `[A-Z]` to match any capital letter followed by `\w*` to match an arbitrary number of alphanumeric characters
```python
# Write the first pattern
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)

# Write the second pattern
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
print(pattern2)

# Write the third pattern
pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
print(pattern3)
```
> True  
> True  
> True

### Custom functions to clean data
* You can use the `.apply()` method to apply a function across entire rows or columns of DataFrames
* However, note that each column of a `DataFrame` is a pandas `Series`. Functions can also be applied across Series
* Here, you will apply your function over the `'sex'` column.
```python
# Define recode_sex()
def recode_sex(sex_value):

    # Return 1 if sex_value is 'Male'
    if sex_value == 'Male':
        return 1
    
    # Return 0 if sex_value is 'Female'    
    elif sex_value == 'Female':
        return 0
    
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
tips['sex_recode'] = tips.sex.apply(recode_sex)

# Print the first five rows of tips
print(tips.head())
```

||total_bill|tip|sex|smoker|day|time|size|sex_recode|
|---|---|-----|------|-----|-----|-----|----|---|
|0|NaN|1.01|Female|No|Sun|Dinner|2.0|0.0|
|1|10.34|1.66|Male|No|Sun|Dinner|3.0|1.0|
|2|NaN|3.50|Male|No|Sun|NaN|3.0|1.0|
|3|23.68|3.31|Male|No|Sun|NaN|2.0|1.0|
|4|24.59|3.61|Female|No|Sun|Dinner|4.0|0.0|

### Lambda functions
*  Your job is to clean its 'total_dollar' column by removing the dollar sign. You'll do this using two different methods:
    * With the `.replace()` method
    * With regular expressions
```python
# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())
```
||total_bill|tip|sex|smoker|day|time|size|total_dollar|total_dollar_replace|total_dollar_re|
|---|-----|----|----|--|---|----|---|---|---|-----|
|0|16.99|1.01|Female|No|Sun|Dinner|2|$16.99|16.99|16.99|  
|1|10.34|1.66|Male|No|Sun|Dinner|3|$10.34|10.34|10.34|  
|2|21.01|3.50|Male|No|Sun|Dinner|3|$21.01|21.01|21.01|  
|3|23.68|3.31|Male|No|Sun|Dinner|2|$23.68|23.68|23.68|  
|4|24.59|3.61|Female|No|Sun|Dinner|4|$24.59|24.59|24.59|

### Dropping duplicate data
```python
# Create the new DataFrame: tracks
tracks = billboard[['year', 'artist', 'track', 'time']]

# Print info of tracks
print(tracks.info())

# Drop the duplicates: tracks_no_duplicates
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks
print(tracks_no_duplicates.info())
```
> <class 'pandas.core.frame.DataFrame'>  
> RangeIndex: 24092 entries, 0 to 24091  
> Data columns (total 4 columns):  
> year      24092 non-null int64  
> artist    24092 non-null object  
> track     24092 non-null object  
> time      24092 non-null object  
> dtypes: int64(1), object(3)  
> memory usage: 753.0+ KB  
> None  

> <class 'pandas.core.frame.DataFrame'>  
> Int64Index: 317 entries, 0 to 316  
> Data columns (total 4 columns):  
> year      317 non-null int64  
> artist    317 non-null object  
> track     317 non-null object  
> time      317 non-null object  
> dtypes: int64(1), object(3)  
> memory usage: 12.4+ KB  
> None  

### Filling missing data
```python
# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality.Ozone.mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality['Ozone'].fillna(oz_mean)

# Print the info of airquality
print(airquality.info())
```
> <class 'pandas.core.frame.DataFrame'>  
> RangeIndex: 153 entries, 0 to 152  
> Data columns (total 6 columns):  
> Ozone      153 non-null float64  
> Solar.R    146 non-null float64  
> Wind       153 non-null float64  
> Temp       153 non-null int64  
> Month      153 non-null int64  
> Day        153 non-null int64  
> dtypes: float64(3), int64(3)  
> memory usage: 7.2 KB  
> None  

### Testing your data with asserts
* To programmatically check for missing values and to confirm that all values are positive
* `all()` method together with the `.notnull()` DataFrame method to check for missing values in a column
* DataFrame needs to chain another .all() method so that you return only one `True` or `False` value
```python
# Assert that there are no missing values in Day column
assert pd.notnul(ebola['Dat']).all()

# Assert that there are no missing values
assert pd.notnull(ebola).all().all()

# Assert that all values are >= 0
assert (ebola >= 0).all().all()
```

## Case Studies
### Exploratory analysis
* The Gapminder data for the 19th century has been loaded into a DataFrame called g1800s
* Use pandas methods such as `.head()`, `.info()`, and `.describe()`, and DataFrame attributes like `.columns` and `.shape` to explore it.
### Visualizing your data
* If points fall on a diagonal line, it means that life expectancy remained the same!
```python
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create the scatter plot
g1800s.plot(kind='scatter', x='1800', y='1899')

# Specify axis labels
plt.xlabel('Life Expectancy by Country in 1800')
plt.ylabel('Life Expectancy by Country in 1899')

# Specify axis limits
plt.xlim(20, 55)
plt.ylim(20, 55)

# Display the plot
plt.show()
```
![alt text](https://g7ylma.ch.files.1drv.com/y4muLo7CcY1zZdiPoSPuUBicmZkGqW6vJoTJFa-8f6722e34jLH2fzUJjKLKITKbrGTJym8Lx9FZ642Y1yga1HrzuMEuzBcJfFbUII6tiSw6LEmgYkOxVacdk8FtiwUSQzmRICd5wB4gdZfk2dys9aVD0m_cKWw6LemfQJznbTUSkgkvM113Gb1rmTn2EpIfFHnjE2TWx1f_WGsPLUQ3IZLzQ/Capture41.png?psid=1)
### Thinking about the question at hand
* How much the average life expectancy changes over each year?
* It's important to make sure that the following assumptions about the data are true:
    * `'Life expectancy'` is the first column (index `0`) of the DataFrame.
    * The other columns contain either null or numeric values.
    * The numeric values are all greater than or equal to 0.
    * There is only one instance of each country.
```python
def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
    no_na = row_data.dropna()[1:-1]
    numeric = pd.to_numeric(no_na, errors='coerce')
    ge0 = numeric >= 0
    return ge0

# Check whether the first column is 'Life expectancy'
assert g1800s.columns[0] == 'Life expectancy'

# Check whether the values in the row are valid
assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# Check that there is only one instance of each country
assert g1800s['Life expectancy'].value_counts()[0] == 1
```

### Assembling your data
* Here, three DataFrames have been pre-loaded: `g1800s`, `g1900s`, and `g2000s`
```python
# Concatenate the DataFrames row-wise
gapminder = pd.concat([g1800s, g1900s, g2000s])

# Print the shape of gapminder
print(gapminder.shape)

# Print the head of gapminder
print(gapminder.head())
```

### Reshaping your data
* What you want is a single column that contains the year, and a single column that represents the average life expectancy for each year and country
```python
# Melt gapminder: gapminder_melt
gapminder_melt = pd.melt(gapminder, id_vars='Life expectancy')

# Rename the columns
gapminder_melt.columns = ['country', 'year', 'life expectancy']

# Print the head of gapminder_melt
print(gapminder_melt.head())
```
||country|year|life expectancy|
|-----|------|-----|-----|
|0|Abkhazia|1800|NaN|
|1|Afghanistan|1800|28.21|
|2|Akrotiri and Dhekelia|1800|NaN|
|3|Albania|1800|35.40|
|4|Algeria|1800|28.82|
### Checking the data types
* Now that your data is in the proper shape, you need to ensure that the columns are of the proper data type
* Need to ensure that `country` is of type `object`, `year` is of type `int64`, and `life_expectancy` is of type `float64`.
> gapminder.info()  
> <class 'pandas.core.frame.DataFrame'>  
> RangeIndex: 169260 entries, 0 to 169259  
> Data columns (total 3 columns):  
> country            169260 non-null object  
> year               169260 non-null object  
> life_expectancy    43857 non-null float64  
> dtypes: float64(1), object(2)  
> memory usage: 3.9+ MB  

```python
# Convert the year column to numeric
gapminder.year = pd.to_numeric(gapminder.year, errors='coerce')

# Test if country is of type object
assert gapminder.country.dtypes == np.object

# Test if year is of type int64
assert gapminder.year.dtypes == np.int64

# Test if life_expectancy is of type float64
assert gapminder.life_expectancy.dtypes == np.float64
```
### Looking at country spellings
* Next task in the data cleaning process is to look at the `'country'` column to see if there are any special or invalid characters you may need to deal with
* Some kinds of special or invalid character:
    * The set of lower and upper case letters.
    * Whitespace between words.
    * Periods for any abbreviations.
* Python has a built-in string method - `str.contains()` - which takes a regular expression pattern, and applies it to the `Series`, returning `True` if there is a match, and `False` otherwise.
```python
# Create the series of countries: countries
countries = gapminder['country']

# Drop all the duplicates from countries
countries = countries.drop_duplicates()

# Write the regular expression: pattern
pattern = '^[A-Za-z\.\s]*$'

# Create the Boolean vector: mask
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse
mask_inverse = ~mask

# Subset countries using mask_inverse: invalid_countries
invalid_countries = countries.loc[mask_inverse]

# Print invalid_countries
print(invalid_countries)
```
|||
|-------|-------------------------|
|49 |           Congo, Dem. Rep.|
|50 |                Congo, Rep.|
|53 |              Cote d'Ivoire|
|73 |     Falkland Is (Malvinas)|
|93 |              Guinea-Bissau|
|98 |           Hong Kong, China|
|118|    United Korea (former)\n|
|131|               Macao, China|
|132|             Macedonia, FYR|
|145|      Micronesia, Fed. Sts.|
|161|            Ngorno-Karabakh|
|187|             St. Barthélemy|
|193|     St.-Pierre-et-Miquelon|
|225|                Timor-Leste|
|251|      Virgin Islands (U.S.)|
|252|       North Yemen (former)|
|253|       South Yemen (former)|
|258|                      Åland|
Name: country, dtype: object
### More data cleaning and processing
* There are several strategies for this:
    * You can drop them
    * Fill them in using the mean of the column or row that the missing value is in (also known as [imputation](https://en.wikipedia.org/wiki/Imputation_(statistics)))
    * If you are dealing with time series data, use a forward fill or backward fill, in which you replace missing values in a column with the most recent known value in the column(mode)
* Drop the rows where any observation in `life_expectancy` is missing. 
* As you confirmed that `country` and `year` don't have missing values, you can use the `.dropna()` method on the entire `gapminder` DataFrame
* The `.dropna()` method has the default keyword arguments `axis=0` and `how='any'`, which specify that rows with any missing values should be dropped.
```python
# Assert that country does not contain any missing values
assert pd.notnull(gapminder.country).all()

# Assert that year does not contain any missing values
assert pd.notnull(gapminder.year).all()

# Drop the missing values
gapminder = gapminder.dropna()

# Print the shape of gapminder
print(gapminder.shape)
```
> (43857, 3)
### Wrapping up
```python
# Add first subplot
plt.subplot(2, 1, 1) 

# Create a histogram of life_expectancy
gapminder.life_expectancy.plot(kind='hist')

# Group gapminder: gapminder_agg
gapminder_agg = gapminder.groupby('year')['life_expectancy'].mean()

# Print the head of gapminder_agg
print(gapminder_agg.head())

# Print the tail of gapminder_agg
print(gapminder_agg.tail())

# Add second subplot
plt.subplot(2, 1, 2)

# Create a line plot of life expectancy per year
gapminder_agg.plot(y='life_expectancy')

# Add title and specify axis labels
plt.title('Life expectancy over the years')
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder.to_csv('gapminder.csv')
gapminder_agg.to_csv('gapminder_agg.csv')
```
![alt text](https://g7wkja.ch.files.1drv.com/y4mFY7uQr3M9VCRkDUY0RzbvwuN5Z6GXgqtsQJiLo7F4wslYG3uonZPT7XruCQ9oYT_258mUruyTe6XkZwKZEFdjXBGfU02uRsPP9IjjvVlSDRlQi2brYHHYydaoUgn96owK2Wyx2Ki4IxwnQiT_7aqdn4RN_XFFQ2qu1XlwRiRP6G3HPaSuDIg6f7mc4m1gKDJfvWj6yRg1EZo-H5vhPYZPA/Capture42.png?psid=1)
