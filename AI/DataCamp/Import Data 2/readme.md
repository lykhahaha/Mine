# Import Data 2
## Importing data from the Internet
### Importing flat files from the web
* The flat file you will import will be `'winequality-red.csv'` from the University of California, Irvine's [Machine Learning repository](http://archive.ics.uci.edu/ml/index.html)
```python
# Import package
from urllib.request import urlretrieve

# Import pandas
import pandas as pd

# Assign url of file: url
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'

# Save file locally
urlretrieve(url, 'winequality-red.csv')

# Read file into a DataFrame and print its head
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())
```

### Opening and reading flat files from the web
* if you just wanted to load a file from the web into a DataFrame without first saving it locally, you can do that easily using `pandas`
```python
# Import packages
import matplotlib.pyplot as plt
import pandas as pd

# Assign url of file: url
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'

# Read file into a DataFrame: df
df = pd.read_csv(url, sep=';')

# Print the head of the DataFrame
print(df.head())

# Plot first column of df
pd.DataFrame.hist(df.ix[:, 0:1])
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()
```
![altt text](https://9txxla.ch.files.1drv.com/y4mENqrrOcm2DoIHIDVtYBfVgwDI82Y0GIdKADTAdNT5b8-oQRB99mWcA6gaOL1oEuMgGiA_fimzKvPTWOAf7ctDZ3ebisyKmjgQ4E5hi3uYKd0n-G2MgrjfgWL3GIv84joYIAdGL7OpVSzQQXzNaZ9Ewsh69G4cY4G9N1ja9JHbLgF6hnDrLEhrw_tSo3ATdZWlm13QiLyyiv8-He-pR70Tg/Capture32.png?psid=1)

### Importing non-flat files from the web
* `pd.read_csv()` allow you to load all types of files, not only flat ones
* In this interactive exercise, you'll use `pd.read_excel()` to import an Excel spreadsheet.
* In order to import all sheets you need to pass `None` to the argument `sheetname`.
```python
# Import package
import pandas as pd

# Assign url of file: url
url = 'http://s3.amazonaws.com/assets.datacamp.com/course/importing_data_into_r/latitude.xls'

# Read in all sheets of Excel file: xl
xl = pd.read_excel(url, sheetname=None)

# Print the sheetnames to the shell
print(xl.keys())

# Print the head of the first sheet (using its name, NOT its index)
print(xl['1700'].head())
```

### Performing HTTP requests in Python using urllib
In this interactive exercise, you will ping our very own DataCamp servers to perform a GET request to extract information from our teach page, `"http://www.datacamp.com/teach/documentation"`
```python
# Import packages
from urllib.request import urlopen, Request

# Specify the url
url = "http://www.datacamp.com/teach/documentation"

# This packages the request: request
request = Request(url)

# Sends the request and catches the response: response
response = urlopen(request)

# Print the datatype of response
print(type(response))

# Be polite and close the response!
response.close()
```
> <class 'http.client.HTTPResponse'>

### Printing HTTP request results in Python using urllib
```python
# Import packages
from urllib.request import urlopen, Request

# Specify the url
url = "http://www.datacamp.com/teach/documentation"

# This packages the request
request = Request(url)

# Sends the request and catches the response: response
response = urlopen(request)

# Extract the response: html
html = response.read()

# Print the html
print(html)

# Be polite and close the response!
response.close()
```
> b'\n<!DOCTYPE HTML>\n<html lang="" >\n    <head>\n        <meta charset="UTF-8">\n        <meta content="text/html; charset=utf-8" http-equiv="Content-Type">\n        <title>Welcome \xc2\xb7 Authoring Content for DataCamp</title>\n        <meta http-equiv="X-UA-Compatible" content="IE=edge" />\n        <meta name="description" content="">\n        <meta name="generator" content="GitBook 3.2.3">\n        \n        \n        \n    \n    <link rel="stylesheet" href="gitbook/style.css">\n\n    \n            \n                \n                <link rel="stylesheet" href="gitbook/gitbook-plugin-bootstrap-callout/plugin.css">\n                \n            \n                \n                <link rel="stylesheet" href="gitbook/gitbook-plugin-expandable-chapters/expandable-chapters.css">\n                \n            \n                \n                <link rel="stylesheet" href="gitbook/gitbook-plugin-advanced-emoji/emoji-website.css">\n                \n            \n                \n 

### Performing HTTP requests in Python using requests
```python
# Import package
import requests

# Specify the url: url
url = 'http://www.datacamp.com/teach/documentation'

# Packages the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response: text
text = r.text

# Print the html
print(text)
```
> <!DOCTYPE HTML>  
> <html lang="" >  
>     <head>  
>         <meta charset="UTF-8">  
>         <meta content="text/html; charset=utf-8"   http-equiv="Content-Type">
>         <title>Welcome · Authoring Content for DataCamp</title>

### Parsing HTML with BeautifulSoup
* You'll scrape the data from the webpage of Guido van Rossum, Python's very own [Benevolent Dictator for Life](https://en.wikipedia.org/wiki/Benevolent_dictator_for_life)
* In the following exercises, you'll prettify the HTML and then extract the text and the hyperlinks.
```python
# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url: url
url = 'https://www.python.org/~guido/'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extracts the response as html: html_doc
html_doc = r.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Prettify the BeautifulSoup object: pretty_soup
pretty_soup = soup.prettify()

# Print the response
print(pretty_soup)
```
> <html>  
> 	<head>  
>  		<title>  
>   Guido's Personal Home Page  
>  		</title>  
> 	</head>  
> 	<body bgcolor="#FFFFFF" text="#000000">  
>   	<h1>  
>   		<a href="pics.html">

### Turning a webpage into data using BeautifulSoup: getting the text
```python
# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url: url
url = 'https://www.python.org/~guido/'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response as html: html_doc
html_doc = r.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Get the title of Guido's webpage: guido_title
guido_title = soup.title

# Print the title of Guido's webpage to the shell
print(guido_title)

# Get Guido's text: guido_text
guido_text = soup.get_text()

# Print Guido's text to the shell
print(guido_text)
```
> <title>Guido's Personal Home Page</title>  
> Guido's Personal Home Page  
> Guido van Rossum - Personal Home Page  
> "Gawky and proud of it."  
> Who  
> I Am  
> Read

### Turning a webpage into data using BeautifulSoup: getting the hyperlinks
```python
# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url
url = 'https://www.python.org/~guido/'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extracts the response as html: html_doc
html_doc = r.text

# create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Print the title of Guido's webpage
print(soup.title)

# Find all 'a' tags (which define hyperlinks): a_tags
a_tags = soup.find_all('a')

# Print the URLs to the shell
for link in a_tags:
    print(link.get('href'))
```
> <title>Guido's Personal Home Page</title>
> pics.html  
> http://www.washingtonpost.com/wp-srv/business/longterm/microsoft/stories/1998/raymond120398.htm  
> http://metalab.unc.edu/Dave/Dr-Fun/df200004/df20000406.jpg  
> http://neopythonic.blogspot.com/2016/04/kings-day-speech.html  
> http://www.python.org  
### Loading and exploring a JSON
```python
# Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])
```
> Genre:  Biography, Drama  
> Country:  USA  
> imdbRating:  7.7  
> DVD:  11 Jan 2011  
> Website:  http://www.thesocialnetwork-movie.com/
### API requests
```python
# Import requests package
import requests

# Assign URL to variable: url
url = 'http://omdbapi.com/?t=social+network&apikey=ff21610b'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Print the text of the response
print(r.text)
```
> {"Title":"The Social Network","Year":"2010","Rated":"PG-13","Released":"01 Oct 2010","Runtime":"120 min","Genre":"Biography, Drama"}
### JSON–from the web to Python
```python
# Import package
import requests

# Assign URL to variable: url
url = 'http://www.omdbapi.com/?apikey=ff21610b&t=social+network'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Decode the JSON data into a dictionary: json_data
json_data = r.json()

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])
```
> Genre:  Biography, Drama  
> Country:  USA  
> Language:  English, French  
> DVD:  11 Jan 2011  
> Website:  http://www.thesocialnetwork-movie.com/
### Checking out the Wikipedia API
* the Wikipedia API (documented [here](https://www.mediawiki.org/wiki/API:Main_page))
* You'll figure out how to find and extract information from the Wikipedia page for Pizza
* The result is nested JSONs, that is, JSONs with JSONs, but Python can handle that because it will translate them into dictionaries within dictionaries.
```python
# Import package
import requests

# Assign URL to variable: url
url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Decode the JSON data into a dictionary: json_data
json_data = r.json()

# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)
```

```html
<p><b>Pizza</b> is a traditional Italian dish consisting of a yeasted flatbread typically topped</p>
<p>The term <i>pizza</i> was first recorded in the 10th century</a></p></p>
```

## Diving deep into the Twitter API
### API Authentication
* The package `tweepy` is great at handling all the Twitter API OAuth Authentication details for you
* All you need to do is pass it your authentication credentials
```python
# Import package
import tweepy

# Store OAuth authentication credentials in relevant variables
access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"
consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"

# Pass OAuth details to tweepy's OAuth handler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
```

### Streaming tweets
* Now that you have set up your authentication credentials, it is time to stream some tweets
```python
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open("tweets.txt", "w")

    def on_status(self, status):
        tweet = status._json
        self.file.write( json.dumps(tweet) + '\n' )
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False
        self.file.close()

    def on_error(self, status):
        print(status)

 # Initialize Stream listener
l = MyStreamListener()

# Create you Stream object with authentication
stream = tweepy.Stream(auth, l)


# Filter Twitter Streams to capture data by the keywords:
stream.filter(track=['clinton', 'trump', 'sanders', 'cruz'])
```
### Load and explore your Twitter data
* Now that you've got your Twitter data sitting locally in a text file, it's time to explore it!
```python
# Import package
import json

# String of path to file: tweets_data_path
tweets_data_path = 'tweets.txt'

# Initialize empty list to store tweets: tweets_data
tweets_data = []

# Open connection to file
tweets_file = open(tweets_data_path, "r")

# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)

# Close connection to file
tweets_file.close()

# Print the keys of the first tweet dict
print(tweets_data[0].keys())
```
> dict_keys(['id_str', 'retweet_count', 'source', 'extended_entities', 'entities', 'in_reply_to_screen_name', 'coordinates', 'favorited', 'text', 'in_reply_to_user_id', 'retweeted', 'truncated'])
### Twitter data to DataFrame
```python
# Import package
import pandas as pd

# Build DataFrame of tweet texts and languages
df = pd.DataFrame(tweets_data, columns=['text', 'lang'])

# Print head of DataFrame
print(df.head())
```
||text|lang|
|--|-------------------------------------|--|
|0|b"RT @bpolitics: .@krollbondrating's Christoph...|en|
|1|b'RT @HeidiAlpine: @dmartosko Cruz video found... |en|
### A little bit of Twitter text analysis
* Do a bit of text analysis to count how many tweets contain the words `'clinton'`, `'trump'`, `'sanders'` and `'cruz'`
* `word_in_text()`, which will tell you whether the first argument (a word) occurs within the 2nd argument (a tweet).
```python
import re

def word_in_text(word, tweet):
    word = word.lower()
    text = tweet.lower()
    match = re.search(word, tweet)

    if match:
        return True
    return False

# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]

# Iterate through df, counting the number of tweets in which
# each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])
```
### Plotting your Twitter data
```python
# Import packages
import seaborn as sns
import matplotlib.pyplot as plt


# Set seaborn style
sns.set(color_codes=True)

# Create a list of labels:cd
cd = ['clinton', 'trump', 'sanders', 'cruz']

# Plot histogram
ax = sns.barplot(cd, [clinton, trump, sanders, cruz])
ax.set(ylabel="count")
plt.show()
```
![alt text](https://9tu8ia.ch.files.1drv.com/y4mdopAT3MFjZiLJy-fZMypciJFCQQ_CtZ-FLt4oMhDPIJnjXK5o3eRPcwGikdY3s2TDUvOmv6dd_nl27ZwOFRHf7Xn-r8g2BYr-0qw7mteSyTWtbEH2Ii954FkOzussvzQ8sHVjJaYkekgGWnofGGXs1AUZ-7dPvXP3dD9Q-WEXiSYPDh3kp22qZAW6vnoa8-IbtODqyaih_UPtW7aJ-OA3g/Capture33.png?psid=1)