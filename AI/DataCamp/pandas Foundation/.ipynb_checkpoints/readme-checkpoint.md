# pandas Foundation
## Data ingestion & inspection
### Inspecting your data
### NumPy and pandas working together
* You can use the DataFrame attribute `.values` to represent a `DataFrame` df as a `NumPy` array
```python
# Import numpy
import numpy as np

# Create array of DataFrame values: np_vals
np_vals = df.values

# Create new array of base 10 logarithm values: np_vals_log10
np_vals_log10 = np.log10(np_vals)

# Create array of new DataFrame by passing df to np.log10(): df_log10
df_log10 = np.log10(df)

# Print original and new data containers
print(type(np_vals), type(np_vals_log10))
print(type(df), type(df_log10))
print(df)
print(df_log10)
```
> <class 'numpy.ndarray'> <class 'numpy.ndarray'>  
> <class 'pandas.core.frame.DataFrame'> <class 'pandas.core.frame.DataFrame'>  
      
|Year|Total Population|
|-----|------------|
|1960|3.034971e+09|
|1970|3.684823e+09|
|1980|4.436590e+09|
|1990|5.282716e+09|
|2000|6.115974e+09|
|2010|6.924283e+09|
      
|Year|Total Population|
|-----|---------------|
|1960|9.482154|
|1970|9.566417|
|1980|9.647049|
|1990|9.722857|
|2000|9.786466|
|2010|9.840375|

### Zip lists to build a DataFrame
```python
print(list_keys)
print(list_values)

# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys, list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)
```
> ['Country', 'Total']  
> [['United States', 'Soviet Union', 'United Kingdom'], [1118, 473, 273]]  
> [('Country', ['United States', 'Soviet Union', 'United Kingdom']), ('Total', [1118, 473, 273])]
          
||Country|Total|
|---|--------|-------|
|0|United States|1118|
|1|Soviet Union|473|
|2|United Kingdom|273|

### Labeling your data
```python
# Build a list of labels: list_labels
list_labels = ['year', 'artist', 'song', 'chart weeks']

# Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels
```
### Building DataFrames with broadcasting
```python
# Make a string with the value 'PA': state
state = 'PA'

# Construct a dictionary: data
data = {'state':state, 'city':cities}

# Construct a DataFrame from dictionary data: df
df = pd.DataFrame(data)

# Print the DataFrame
print(df)
```
||city|state|
|---|---------|--|
|0 |Manheim|PA|
|1 |Preston park|PA|
|2 |Biglerville|PA|
|3 |Indiana|PA|
|4 |Curwensville|PA|
|5 |Crown|PA|
|6 |Harveys lake|PA|
|7 |Mineral springs|PA|
|8 |Cassville|PA|
|9 |Hannastown|PA|
|10|Saltsburg|PA|
|11|Tunkhannock|PA|
|12|Pittsburgh|PA|
|13|Lemasters|PA|
|14|Great bend|PA|

![alt text](https://g7wd3q.ch.files.1drv.com/y4myQylC_d3bnjcn6CaQV6dVi_zx8Hob0yc6vrKtwKYI01Ab5eAh3i_SbnT49SeQXYT9DnOqTHLrzVQH0yJM6p36PATwUZ69JQNpYeqWZWYefk3l9SzJSIGoKpcFZ6Z7a_Clh66gNBeLgHdQsFpcM3Fmk-K1cl1zt5Yq9Hrk0GsW_Assn9Xg2bJ-zvWSL78B1bemainNkMOM6xM8o_fe6yzmQ/Capture43.PNG?psid=1)

![alt text](https://g7xoaa.ch.files.1drv.com/y4m7fHnUiswFwbNeg7I02zRd4iu2JQl6Lrx46Tt5A0GAr-EqBjkvr_Tpo_y9nrvGfDF8XUMRQklT-aSyceDi7SnZaWPaivtuwSTljykq0zeKNSgP4_MPC9-QfOPBVhYZBGe5avnWXAGAB2Qx7Zvj96DsvflVFGd78HJrVBt0qnqjRbvLtz1Be2dp1_gn2OtprnSud7olGphPd79BnYpUmDJmw/Capture44.PNG?psid=1)

![alt text](https://g7xoaa.ch.files.1drv.com/y4m7fHnUiswFwbNeg7I02zRd4iu2JQl6Lrx46Tt5A0GAr-EqBjkvr_Tpo_y9nrvGfDF8XUMRQklT-aSyceDi7SnZaWPaivtuwSTljykq0zeKNSgP4_MPC9-QfOPBVhYZBGe5avnWXAGAB2Qx7Zvj96DsvflVFGd78HJrVBt0qnqjRbvLtz1Be2dp1_gn2OtprnSud7olGphPd79BnYpUmDJmw/Capture44.PNG?psid=1)

### Reading a flat file
```python
# Read in the file: df1
df1 = pd.read_csv('world_population.csv')

# Create a list of the new column labels: new_labels
new_labels = ['year', 'population']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv('world_population.csv', header=0, names=new_labels)

# Print both the DataFrames
print(df1)
print(df2)
```
||Year|Total Population|
|--|------|--------|
|0|1960|3.034971e+09|
|1|1970|3.684823e+09|
|2|1980|4.436590e+09|
|3|1990|5.282716e+09|
|4|2000|6.115974e+09|
|5|2010|6.924283e+09|

||year|population|
|-|-----|------------|
|0|1960|3.034971e+09|
|1|1970|3.684823e+09|
|2|1980|4.436590e+09|
|3|1990|5.282716e+09|
|4|2000|6.115974e+09|
|5|2010|6.924283e+09|

### Delimiters, headers, and extensions
* The file name is given to you in the variable file_messy
*  This file has three aspects that may cause trouble for lesser tools: 
    * multiple header lines
    * comment records (rows) interleaved throughout the data rows 
    * tab delimiters instead of commas.
```python
# Read the raw file as-is: df1
df1 = pd.read_csv(file_messy)

# Print the output of df1.head()
print(df1.head())

# Read in the file with the correct parameters: df2
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')

# Print the output of df2.head()
print(df2.head())

# Save the cleaned up DataFrame to a CSV file without the index
df2.to_csv(file_clean, index=False)

# Save the cleaned up DataFrame to an excel file without the index
df2.to_excel('file_clean.xlsx', index=False)
```
> The following stock data was collect on 2016-AUG-25 from an unknown source  
> These kind of ocmments are not very useful, are they?  
> probably should just throw this line away too, but not the next since those are column labels  
> name Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec  
> `# So that line you just read has all the column headers labels`  
> IBM 156.08 160.01 159.81 165.22 172.25 167.15 164.75 152.77 145.36 146.11 137.21 137.96

||name|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|
|---|-----|-----|------|-----|-----|-----|------|-----|-----|-----|-----|------|-----|
|0|IBM|156.08|  160.01|  159.81|  165.22| 172.25|167.15|164.75|152.77|   145.3|  146.11|  137.21|  137.96|  
|1|MSFT|45.51|   43.08|   42.13|   43.47|  47.53| 45.96| 45.61| 45.51|    43.5|   48.70|   53.88|   55.40|  
|2|GOOGLE|512.42|  537.99|  559.72|  540.50| 535.24|532.92|590.09|636.84|   617.9|  663.59|  735.39|  755.35|  
|3|APPLE|110.64|  125.43|  125.97|  127.29| 128.76|127.81|125.34|113.39|   112.8|  113.36|  118.16|  111.73|

### Plotting series using pandas
![alt text](https://g7yeuq.ch.files.1drv.com/y4mVO7I_j4nMSl0uAV-nF3dVHna5ykOmirNC2RNF3Fwn5p6Q-J3RgAtLshpzKQ-1t6NrflX_ldTkgCLcdhbqPFga4_Vx0jwl5jP3Rb4F5tWVPdXXtWt9Alef-GJzdE5Yx3X9M3lDog0C8CRwIDBIqZKBBNIH6lHqy1n4WPsSt-7WQyrNnrCrznir1sstMHyG97WuD7_mniZylZvnFMTJkDnqg/Capture46.PNG?psid=1)
![alt text](https://g7z8lq.ch.files.1drv.com/y4mduQFwv5KybngWhEhjz52q4jR2nUl-Ukb9w5ybfnsOX59VNxip2j4hE6it0VWMEa3askig-8JWyrtjW3nAtEvMyYBRi7B0x3Jt3xqgSkIHbYw4SHw7X0R9rJYGms6FRX4dqRSCUWGL5aAuY3v2N0_ZHxIbjBaJkS8WReRc1vhupO_zsP2f5yPUDGUuE1gB1WeRYdOELM5EB4xBDjL2BUrxg/Capture47.PNG?psid=1)
![alt text](https://g7yxcg.ch.files.1drv.com/y4mc-OzaWt5pkhHjo3Gl0hYOs5YaWVTiX7dlNVek2rxKqam_jr3Q24WqlXpYw2w5CsyyBThT-7OHuVH_igHB1EnfVJnK5Y7LsH8YT2iJhy7USoYras41NydF1AsfOBopFzcct30kC64rBZK6l0bbL8ic3A0ZFXnkXw_UZ9La-QBDR-6IwJaTvtwwTAZG52q-05hiFxvuoGOOF6jBUP5D2khuA/Capture48.PNG?psid=1)
![alt text](https://vbrlig.ch.files.1drv.com/y4mkssXAxrgWzVi6ZOHOWBLLKshB_G4lvcfqx_46pPos1YjQUG_djpG9BXKIpQYOh8UxiNfQZ1DH5hu7X9DHdpyv8MCKrndjUDnxEuCL4roOuOMVGqQlCQhFAmjkzOpr74YYQ2ddWgur1UFRc0SWiS4sjhfLnF_8nHs1SDGJtd21UaXZb-F9XvQsTIjalrXUQ7G1NdnxrIB-Z2U575bYUe1yA/Capture49.PNG?psid=1)
![alt text](https://vbqcga.ch.files.1drv.com/y4m0NPJePx-Qe9hx0rz-H5PD0O8DmDoq4LJVjdJoEwhdYU-u1t9huShgYfD85_XXt3ri1Ei604FY-3a36ffc14S_7W1aVWCz-dQtChCWuJlZAhIlI64yqp3aDDlHvG217IJJo4PqkQRVrBSmdYmCP18UEPQDd0L-nae0wsOHUpe0GL_1OqgusG7tx1mDmdhKQxag8Mv58P4S54GqdzQxR0kGg/Capture50.PNG?psid=1)
![alt text](https://vbpmlw.ch.files.1drv.com/y4m0NZLWVCqbvWdO5D6XXjtQgRNdRVCoXyQ38cr8kRExg0gCkzxm9xtdAdQQYBO_gycCdzkYp2eTUn7EaCyyeB30oKwkGtv4CqmBEtouni56emZCuw7e5j5Yz4ya3PHecITLhPpDDIuzHJa8_lqxZaE_Eq6ruFCAoGVvs7-FDU8-xUMN_EPMVAJV3grH8apdwcxwVLdRF2Y5J-V3LmhOElt_A/Capture51.PNG?psid=1)
> plt.head()

||Temperature (deg F)|
|---|------|
|0|79.0|
|1|77.4|
|2|76.4|
|3|75.7|
|4|75.1|

```python
# Create a plot with color='red'
df.plot(color='red')

# Add a title
plt.title('Temperature in Austin')

# Specify the x-axis label
plt.xlabel('Hours since midnight August 1, 2010')

# Specify the y-axis label
plt.ylabel('Temperature (degrees F)')

# Display the plot
plt.show()
```
![alt text](https://vbobda.ch.files.1drv.com/y4mJjKggxxuFd8AFQLtBwD9LoIdLRWGJK6Wvy2L4Fzb6BCy6npefkvGN-gt7iqstmz2Df9qIXr2pbzTWVvUR-ZOTIr6t8bKllvWCCNUg85GfwZEs_hBs27zsO7BdRWuSgIagrJqpFgpFeekUX0l0WJ9K1br2WSFrJ7luQW8mu1H3tKn4znK670IVVmVv6rqbXwwfWEsyjB8iFB8sLq-pl_-hA/Capture52.png?psid=1)
### Plotting DataFrame
* Pre-loaded three columns of data from a weather data set - temperature, dew point, and pressure
* However, measured in `Atmospheres`, has a different vertical scaling than that of the other two data columns, which are both measured in degrees `Fahrenheit`
```python
# Plot all columns (default)
df.plot()
plt.show()

# Plot all columns as subplots
df.plot(subplots=True)
plt.show()

# Plot just the Dew Point data
column_list1 = ['Dew Point (deg F)']
df[column_list1].plot()
plt.show()

# Plot the Dew Point and Temperature data, but not the Pressure data
column_list2 = ['Temperature (deg F)','Dew Point (deg F)']
df[column_list2].plot()
plt.show()
```
![alt text](https://vbpfua.ch.files.1drv.com/y4mhXmLEKB0feqzE0yKt24csKUznOjINNS3WmEYW-i7VGFgx6nOV9G-RZUmtZJ7KhYkjrzOqCL6XVyi7oYtHBmvhwsRHZinXgZnnAy0fXXQOks5Ovn3nu-rqpGf0d9NraoY9Ciue6i-boSg6MxBEj3GYIsTJRFwVHtvTjZV5sp5f0WDGu5zJsFkP6cqeSrk1_jCnSIh58-pG6ZbKyymrcHV3A/Capture53.png?psid=1)
![alt text](https://vbr6xa.ch.files.1drv.com/y4mkBjymMNNxByx4jJXxKv8hwD2yL2LEd_rxmFkKI5IN1EQ9p6_-Db81dWi9LxsQTByZqT6sPrQ7e4nPi2f7dLcVAcoFu8DkIcGe3r8JPXewtijp-vHs-QmAdpJMbY7IbvAiU-Sx3UX3hPzCp98BVdCjZGHzaCiHoho792SB-KPaaoO7cifZUY3B05-WHXVy0Bb4SfTcljLB7zL-iXhTfvXCg/Capture54.png?psid=1)
![alt text](https://vbowrg.ch.files.1drv.com/y4mGWbYFWnzU8YSPx401M9OBxFIW5I_06qq1LnouwHa0WBupfXo62HRJadIxyasUsbiCeq9O2D5yuHy2hbrR2OuDtkaHVvCpIJ-70WgmsbcBZ5nOVztGc5kjKEYiRLfYeFX-xGYJBX-YM5-X028mTTXGO19jc6ZOr6UElYwGSttauzF_yFNX5U-GcMU5bQVKFNpY5s4sx08GNpXDEUWRyC9Ew/Capture55.png?psid=1)
![alt text](https://vbqvoq.ch.files.1drv.com/y4mPRk0LyFCQ-xPVQp6cPK5-mlD1ObRyBenlhY3-iTmxWoQv_yA8vEhAdfgtsgOPjUSvwtwpWPCFHuMjuZrkHND6aDqNc5pfv4l7o26PoD9aSCf-0TKEtbBh0oSaQCBP9OB2Q3Qi9MgtlCXjDAIAWr0hd5zozQG5RcvPFmAOubhEmbyYDyFzmnyE1xacUdAFFvPSYmUJjf8eHO8_mRLxyZXgQ/Capture56.png?psid=1)

## Exploratory data analysis
### pandas line plots
```python
# Create a list of y-axis column names: y_columns
y_columns = ['AAPL', 'IBM']

# Generate a line plot
df.plot(x='Month', y=y_columns)

# Add the title
plt.title('Monthly stock prices')

# Add the y-axis label
plt.ylabel('Price ($US)')

# Display the plot
plt.show()
```
![alt text](https://vbqo8g.ch.files.1drv.com/y4m2QOdB9E1FqXNPYJmrVse7zSgh80xRlNGC9_UXAwJeScfo_F4v016nqQ2_4rM90EEsNb4YJNJE3TRYN5-z3nuaqxdR-EBr1oL78ZQPT6r2dp1ojD1Kb9zDgZcglbxysJH7zmKmOMuu0r0--YW3HzjmtY4XN_hFEr0aL3ZAZwKSVWYJtUYCXUTF7OZF7ZR_3lE_v517vM8bmLm2_LAccwjfQ/Capture57.png?psid=1)

### pandas scatter plots
* Dataset: 'auto-mpg.csv'
* The size of each circle is provided as a NumPy array called `sizes`. This array contains the normalized `'weight'` of each automobile in the dataset.
> sizes  
> 51.12044694  56.78387977  49.15557238  49.06977358  49.52823321
```python
# Generate a scatter plot
df.plot(kind='scatter', x='hp', y='mpg', s=sizes)

# Add the title
plt.title('Fuel efficiency vs Horse-power')

# Add the x-axis label
plt.xlabel('Horse-power')

# Add the y-axis label
plt.ylabel('Fuel efficiency (mpg)')

# Display the plot
plt.show()
```
![alt text](https://vbrzfq.ch.files.1drv.com/y4miPpWkPzscWzPDU8KvIkoiVRYa4_0u8bDhAW_PZJIuX7Vb1rqOXiHfuZiADiCHZPQVa8wAc8AATDk80wcDJKbsjqWyuD4h8lb4LbsCXQU-PblE3zt23qGewt7xRY-AR5xS20PIW7zhiGU-kZpfcjvKe2mRKEyvJ0V14gtbLvMk32O7jpiRncAE4R2w1NFhiiP_Qb3Mrz69yP7AcBIe8S4sA/Capture58.png?psid=1)

### pandas subplot box plots
* `y=cols` is for **box plots**
```python
# Make a list of the column names to be plotted: cols
cols = ['weight', 'mpg']

# Generate the box plots
df.plot(kind='box', y=cols, subplots=True)

# Display the plot
plt.show()
```
![alt text](https://uawp9a.ch.files.1drv.com/y4m7IQUQdr7-CsTZGouTjUzvH-3qP6FAd7kxcpcwYuNgMUWzuo4IWmnFzgbLLb2J1eopryqIaiKcR4m-MPXhiX4Ehrcfg_IaWyxR9mcvd-3XAE4SaHLUODHeZOgASTtFediZvKAd4-24PqdwJAzbWm1VUPSLAzHdrBmiZAI6xGr2QPGcFPz0dVvnSAJ3qm4jKu_BY7UUvA3a2gX1k0vzSwuYQ/Capture59.png?psid=1)

### pandas hist, pdf and cdf
* Remember, when plotting the PDF, you need to specify `normed=True` in your call to `.hist()`, and when plotting the CDF, you need to specify `cumulative=True` in addition to normed=True.
```python
# This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)

# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3))
plt.show()

# Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', normed=True, bins=30, cumulative=True, range=(0,.3))
plt.show()
```
![alt text](https://uax03q.ch.files.1drv.com/y4mgobFu31700pzhEUNXPDe6tEc3wVK6ULvUQx17TeH39E_713dn0rs7Qk5i2NC6oLKwtDkKtFrdpeRHWzhIKpeJXqTPWZlKhIUfk4LCcFRaTmHwwY7v-n69vtH0ZaXdN8IhV_rj8IfkuNHz69cT74HzLEhH6QMKbiB3IiDGZIpR6lAFcMf4_NaG18vvOa1YLTgFWmEas1H8lOamNJW0xcekg/Capture60.png?psid=1)

### Bachelor's degrees awarded to women
* `axis='columns'` means plot the `index` dataset
```python
# Print the minimum value of the Engineering column
print(df['Engineering'].min())

# Print the maximum value of the Engineering column
print(df['Engineering'].max())

# Construct the mean percentage per year: mean
mean = df.mean(axis='columns')

# Plot the average percentage per year
mean.plot()

# Display the plot
plt.show()
```
![alt text](https://uavzaq.ch.files.1drv.com/y4mLbs9k3BobmxyIr0OCBNTPhnje1OSpFwQKb7e3yduWW2AU84s6-_pp6CVBCZHL5L68YfQxR1WAL4pPGIsN87JYYJHluAlQtUbEL9cFZhRFQjFvvwqnKu1Sppmgm8_Zzl4LLUx0xy3a-EK9sCS4SwSr5AEML1yeGfAa_cCXytGaYCnJ0zT-AU7atuAtF6gLDYCYLTYF8oJH81klstSXeL1Yg/Capture61.png?psid=1)

### Median vs mean
```python
# Print summary statistics of the fare column with .describe()
print(df.fare.describe())

# Generate a box plot of the fare column
df.fare.plot(kind='box')

# Show the plot
plt.show()
```
> count    1308.000000  
> mean       33.295479  
> std        51.758668  
> min         0.000000  
> 25%         7.895800  
> 50%        14.454200  
> 75%        31.275000  
> max       512.329200

![alt text](https://uavsig.ch.files.1drv.com/y4mLGahdAFhB9UVrkdOUIrPlabV9xtJUWCW5SDpVpNdF7RO8p8fIl-23eQR6lEORB0Nvx3Gx-A-s-Rn9v0bngIM5yxzUdc4qkN6XcMKRXyIyQwcvqOHWxxRBTROP5F4ZxogXnt9J-qdz-YWJRAhsP7VQv7hjBLh7BLvAmUFt3sxunksyRtr6cgtwygQpV-RgebREuSTTdqszWs-12y45SInQQ/Capture62.png?psid=1)

### Quantiles
```python
# Print the number of countries reported in 2015
print(df['2015'].count())

# Print the 5th and 95th percentiles
print(df.quantile([0.05, 0.95]))

# Generate a box plot
years = ['1800','1850','1900','1950','2000']
df[years].plot(kind='box')
plt.show()
```
> 208  

|Unnamed:|   1800 |  1801 |  1802 | 1803  |1804  | 1805 |  1806 |  1807 | 1808  |
|---|----|----|----|----|----|----|----|----|-----|
|0.05    |   12.95|  25.40|  25.30|  25.20|  25.2|  25.2|  25.40|  25.40|  25.40|
|0.95    |  246.05|  37.92|  37.35|  38.37|  38.0|  38.3|  38.37|  38.37|  38.37|

![alt text](https://uaw3rq.ch.files.1drv.com/y4m1w4H8UJDMewGkbDkVAB9RWRnUyH21AOoIHHyAbrYj7_hj1QrwV4EymDl2dvSZV-lECCzSA8CTPh2oSw8bFhqKpQzNX1XCXlpU3Z8jGQ3V-Pf2fqrL25KC73edTOphTY-2Q3QrqTVh-M89Mj24t_HjvW5O_TjwAp1xohPlT7HPAevyCt94SDfIMfMhbbHIgBPtEiPXn8DuuP4e6AK-ovUYw/Capture63.png?psid=1)

### Standard deviation of temperature
> january.head() 

||Date|Temperature|
|---|--------|---|
|0|2013-01-01|28|
|1|2013-01-02|21|
|2|2013-01-03|24|
|3|2013-01-04|28|
|4|2013-01-05|30|
```python
# Print the mean of the January and March data
print(january.mean(), march.mean())

# Print the standard deviation of the January and March data
print(january.std(), march.std())
```
> Temperature    32.354839    
> Temperature    32.354839  
> Temperature    13.583196  
> Temperature    7.478859

### Separate and summarize
```python
# Compute the global mean and global standard deviation: global_mean, global_std
global_mean = df.mean()
global_std = df.std()

# Filter the US population from the origin column: us
us = df[df['origin']=='US']

# Compute the US mean and US standard deviation: us_mean, us_std
us_mean = us.mean()
us_std = us.std()

# Print the differences
print(us_mean - global_mean)
print(us_std - global_std)
```
> mpg        -3.412449  
> cyl         0.805612  
> displ      53.100255  
> hp         14.579592  
> weight    394.905612  
> accel      -0.551122  
> yr         -0.387755  
dtype: float64

> mpg       -1.364623  
> cyl       -0.049788  
> displ     -6.267657  
> hp         1.406630  
> weight   -54.055870  
> accel     -0.022844  
> yr        -0.023369  
dtype: float64

### Separate and plot
```python
# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=3, ncols=1)

# Generate a box plot of the fare prices for the First passenger class
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')

# Generate a box plot of the fare prices for the Second passenger class
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')

# Generate a box plot of the fare prices for the Third passenger class
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')

# Display the plot
plt.show()
```
![alt ext](https://uawicw.ch.files.1drv.com/y4mke3rOakkD5t-rqGucAL5lDHHzvv5_VJ2UrVwiah9jnnSGcsctCyGtfmDZyIqDoq2SHA8iWHLi3wGyDQ5afYJ_hfht6QLBXNGq3YXdFlzcHRv3JHpxEHEL7fan0SiKzQoqAOOZ_1v4eLumn5RnhPxkmKJwvOA5p2zH2FgOYlvNYXBerb7W8FZZC_88c2tSd4iha7JH47_rs_qYK3bsOzATQ/Capture64.png?psid=1)

## Time series in pandas
### Reading and slicing times
* There is difference between `df2 = pd.read_csv(filename, parse_dates=['Date'])` and `df3 = pd.read_csv(filename, index_col='Date', parse_dates=True)`

### Creating and using a DatetimeIndex
* If passed the list of strings `['2015-01-01 091234','2015-01-01 091234']` and a `format` specification variable, such as `format='%Y-%m-%d %H%M%S'`
> date_list[:5]  
> '20100101 00:00',  
> '20100101 01:00',  
> '20100101 02:00',  
> '20100101 03:00',  
> '20100101 04:00'

```python
# Prepare a format string: time_format
time_format = '%Y-%m-%d %H:%M'

# Convert date_list into a datetime object: my_datetimes
my_datetimes = pd.to_datetime(date_list, format=time_format)  

# Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index=my_datetimes)
time_series.head()
```
|||
|---------------------|----|
|2010-01-01 00:00:00|46.2|
|2010-01-01 01:00:00|44.6|
|2010-01-01 02:00:00|44.1|
|2010-01-01 03:00:00|43.8|
|2010-01-01 04:00:00|43.5|

### Partial string indexing and slicing
```python
# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00']

# Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc['2010-07-04']

# Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0['2010-12-15':'2010-12-31']
```
### Reindexing the Index
![alt text](https://uaxtlg.ch.files.1drv.com/y4mqwnpa308No39quSsy6h8-9aamlcNMMyo__9H1zsbGLxKF7YPzhMfhvPfbqk7vx92gROf3RiGcN1oVQ2tHDcYxPqFT3C--CK-gay-UM1kNMjbi-VtmS5SHoF-TAN1giRB6hCE_LRldvvQjoSEVBxN9BC0uiFDfr8FICycsf9--q41sB02LEBHWUWVQfTCB5FcR3HXYEAUk_FtNY38mZvHiw/Capture65.PNG?psid=1)
![alt text](https://uavl2w.ch.files.1drv.com/y4mygYw9T6wo-a4F_24nnfKIIqMtVCDpUMAvnf3UV1Jm_-9f7IINSj-r8TTYajeySWwL9D1gzx7hOQeCNIg5OnaHdudimOoXlcIaOavcl0mE-jX0dXIB7qllyv9WKgqVieSnlFp9O4heyfL4XtqDTQdvzASe0MUTtO9WLdhmKOfgwtCR3h0d-hcdydiSp3swRFgkqRfCKZPNAvN1-C86Athyg/Capture66.PNG?psid=1)
![alt text](https://uaxmtw.ch.files.1drv.com/y4mCrf20UpK2VBqbnpBeW7-wE6BxReMMjGWwEoRG4rY7Wtaj_U_1Kpxn3FfHtPcGlKoRZmg1ZVkSboFJc7aNq8Bh_H3sJtxCBF1CYugyojd8L1zdHL6rEGN4VP2pLg7n4t93z6vSzWzhOsZMHKBj7fi8OT7gAHEUagC0dOf_mrTd_UxJ4HrTM9JITF9SaNquvzajGDG7T8dMlBU37LcWhG5UA/Capture67.PNG?psid=1)
![alt text](https://igo13a.ch.files.1drv.com/y4mCyIZbquOrdEPmd27jLaz_3s0kRQkwBJhBgfDmNFx586BDC9_ZesRsEzlGmGt38xqeD5bgLZE435RbjoUEwPiWd62JAN8tm5W26mZKuTdYKH-MB-qBRJEIKrqOVFz61tSq5mCYaTlg6xEmaAVEtOKbXuQejWlxE7LqdL3ZsDtGxgdgmULgjiR8UgWMYxdrYEc46jyTTH5Rm2gZGppsRXODA/Capture68.PNG?psid=1)