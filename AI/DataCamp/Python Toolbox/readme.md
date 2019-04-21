# Python Toolbox
## Summary
* `global`: make `num = 6` accessable in main
```python
def func2():
    global num
    num = 6
```
* `nonlocal` in Python3: `nonlocal num` must need `echo_word` declare outside its closest `def`
```python
def echo_shout(word):
    echo_word = word
    def shout():
        nonlocal echo_word
        
        echo_word = word + '!!!'
        
    shout()
    
    print(echo_word)
    
echo_shout('hello')
```
> hello!!!
* Nested function: 2 kinds:
```python
def three_shouts(word1, word2, word3):
    def inner(word):
        return word + '!!!'
    return (inner(word1), inner(word2), inner(word3))

print(three_shouts('a', 'b', 'c'))
```
> ('a!!!', 'b!!!', 'c!!!')
```python
def echo(n):
    def inner_echo(word):
        echo_word = word * n
        return echo_word
    return inner_echo

twice = echo(2)
print(twice('hello'))	
```
> hellohello
* Default and flexiable arguments
	* Functions with one default argument: `def shout_echo(word1, echo=1)`
	* Functions with multiple default arguments: `def shout_echo(word1, echo=1, intense=False)`
	* Function with variable-length arguments (`*arg`): `def gibberish(*args): for word in args:`
	* Function with variable-length keyword arguments (`**kwargs`): `def report_status(** kwargs): for key, value in kwargs.items():`
* Lambda functions:
	* Origin lambda: 
```python
    echo_word = (lambda word1, echo: word1*echo)
    result = echo_word('hey', 5)
```
	* `Map()` with list `spells`: 
```python
    shout_spells = map(lambda spell: spell + '!!!', spells)
    print(list(shout_spells))
```
	> ['protego!!!', 'accio!!!', 'expecto patronum!!!', 'legilimens!!!']
	* `Filter()`: 
```python
    len_spell = filter(lambda spell: len(spell) > 6, spells)
    print(list(len_spell))
```
	> ['protego', 'expecto patronum', 'legilimens']
	* `Reduce()`: 
```python
    import functools as ft
    concat_spell = ft.reduce(lambda speel1, speel2: speel1 + speel2, spells)
    print(concat_spell)
```
	> protegoaccioexpecto patronumlegilimens
* Error handling: 2 kinds:
	* `try-except`: 
```python
    def shout_echo(word, echo):
    try:
        echo_word = word * echo
    except:
        print('word must be string')
    return echo_word
    print(shout_echo('hello', 'accelerate'))
```
	> word must be string
	* `if-raise ValueError`: 
```python
    def shout_echo(word, echo):
    if echo < 0:
        raise ValueError('echo must be greater than 0')
    try:
        echo_word = word * echo
    except:
        print('word must be string')
    return echo_word
    print(shout_echo('hello', -2))
```
	> ValueError: echo must be greater than 0

## Writing your own functions
### `+` and `*` with strings  
Unlike with numeric types such as `ints` and `floats`, the `+` operator _concatenates_ strings together, while the `*` _concatenates_ multiple copies of a string together  
### Write a simple function
```python
# Define the function shout
def shout():
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = 'congratulations' + '!!!'

    # Print shout_word
    print(shout_word)

# Call shout
shout()
```
### Single-parameter functions
```python
# Define shout with the parameter, word
def shout(word):
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'

    # Print shout_word
    print(shout_word)

# Call shout with the string 'congratulations'
shout('congratulations')
```
### Functions that return single values
```python
# Define shout with the parameter, word
def shout(word):
    """Return a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'

    # Replace print with return
    return shout_word

# Pass 'congratulations' to shout: yell
yell = shout('congratulations')

# Print yell
print(yell)
```
### Functions with multiple parameters
```python
# Define shout with parameters word1 and word2
def shout(word1, word2):
    """Concatenate strings with three exclamation marks"""
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'
    
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
    
    # Concatenate shout1 with shout2: new_shout
    new_shout = shout1 + shout2

    # Return new_shout
    return new_shout

# Pass 'congratulations' and 'you' to shout(): yell
yell = shout('congratulations', 'you')

# Print yell
print(yell)
```
### A brief introduction to tuples
```python
# Unpack nums into num1, num2, and num3
num1, num2, num3 = nums

# Construct even_nums
num1 = 2
even_nums = (num1, num2, num3)
print(even_nums)
```
> (2, 4, 6)  
### Function that return multiple values
```python
# Define shout_all with parameters word1 and word2
def shout_all(word1, word2):
    
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'
    
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
    
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words = (shout1, shout2)

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1, yell2 = shout_all('congratulations', 'you')

# Print yell1 and yell2
print(yell1)
print(yell2)
```
> congratulations!!!  
> you!!!  
### Bringing it all together (1)  
For this exercise, your goal is to recall how to load a dataset into a DataFrame. The dataset contains Twitter data and you will iterate over entries in a column to build a dictionary in which the keys are the names of languages and the values are the number of tweets in the given language. The file `tweets.csv` is available in your current directory.  
```python
# Import pandas
import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

for entry in col:

    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] = langs_count[entry] + 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)
```
> {'und': 2, 'en': 97, 'et': 1}
### Bringing it all together (2)
```python
# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of 
    occurrences as value for each key."""

    # Initialize an empty dictionary: langs_count
    langs_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over lang column in DataFrame
    for entry in col:

        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry] = langs_count[entry] + 1
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry] = 1

    # Return the langs_count dictionary
    return langs_count

# Call count_entries(): result
result = count_entries(tweets_df, 'lang')

# Print the result
print(result)
```
## Default arguments, variable-length arguments and scope
### Pop quiz on understanding scope
```python
def func1():
    num = 3
    print(num)

def func2():
    global num
    double_num = num * 2
    num = 6
    print(double_num)
```
> `func1()` prints out `3`, `func2()` prints out `10`, and the value of `num` in the global scope is `6`.
* `global num` will make `num = 6` global
### The keyword global
```python
# Create a string: team
team = "teen titans"

# Define change_team()
def change_team():
    """Change the value of the global variable team."""

    # Use team in global scope
    global team

    # Change the value of team in global: team
    team = 'justice league'
# Print team
print(team)

# Call change_team()
change_team()

# Print team
print(team)
```
> teen titans  
> justice league  

### Nested Functions I
```python
# Define three_shouts
def three_shouts(word1, word2, word3):
    """Returns a tuple of strings
    concatenated with '!!!'."""

    # Define inner
    def inner(word):
        """Returns a string concatenated with '!!!'."""
        return word + '!!!'

    # Return a tuple of strings
    return (inner(word1),inner(word2), inner(word3))

# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))
```
### Nested Functions II
One other pretty cool reason for nesting functions is the idea of a closure. This means that the nested or inner function remembers the state of its enclosing scope when called. Thus, anything defined locally in the enclosing scope is available to the inner function even when the outer function has finished execution.  
```python
# Define echo
def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo

# Call echo: twice
twice = echo(2)

# Call echo: thrice
thrice = echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))
```
> hellohello  
> hellohellohello
## Nested functions
### The keyword nonlocal(Python3) and nested functions
```python
# Define echo_shout()
def echo_shout(word):
    """Change the value of a nonlocal variable"""
    
    # Concatenate word with itself: echo_word
    echo_word = word + word
    
    #Print echo_word
    print(echo_word)
    
    # Define inner function shout()
    def shout():
        """Alter a variable in the enclosing scope"""    
        #Use echo_word in nonlocal scope
        nonlocal echo_word
        
        #Change echo_word to echo_word concatenated with '!!!'
        echo_word = echo_word + '!!!'
    
    # Call function shout()
    shout()
    
    #Print echo_word
    print(echo_word)

#Call function echo_shout() with argument 'hello'    
echo_shout('hello')
```
> hellohello  
> hellohello!!!
* Unlike `global` which we declare `global num` and `num = 6` inside `def`, don't need `num` declare in main and `num` can be accessed in main, `nonlocal num` must need `echo_word` declare outside its closest `def`
```python
def echo_shout(word):
    echo_word = word
    def shout():
        nonlocal echo_word
        
        echo_word = word + '!!!'
        
    shout()
    
    print(echo_word)
    
echo_shout('hello')
```
> hello!!!
```python
def echo_shout(word):
    def shout():
        nonlocal echo_word
        
        echo_word = word + '!!!'
        
    shout()
    
    print(echo_word)
    
echo_shout('hello')
```
> SyntaxError: no binding for nonlocal 'echo_word' found
## Default and flexible arguments
### Functions with one default argument
```python
# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
     exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word

# Call shout_echo() with "Hey": no_echo
no_echo = shout_echo('Hey')

# Call shout_echo() with "Hey" and echo=5: with_echo
with_echo = shout_echo('Hey', 5)

# Print no_echo and with_echo
print(no_echo)
print(with_echo)
```
> Hey!!!  
> HeyHeyHeyHeyHey!!!
### Functions with multiple default arguments
```python
# Define shout_echo
def shout_echo(word1, echo=1, intense=False):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Capitalize echo_word if intense is True
    if intense is True:
        # Capitalize and concatenate '!!!': echo_word_new
        echo_word_new = (echo_word).upper() + '!!!'
    else:
        # Concatenate '!!!' to echo_word: echo_word_new
        echo_word_new = echo_word + '!!!'

    # Return echo_word_new
    return echo_word_new

# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
with_big_echo = shout_echo('Hey', 5, True)

# Call shout_echo() with "Hey" and intense=True: big_no_echo
big_no_echo = shout_echo('Hey', True)

# Print values
print(with_big_echo)
print(big_no_echo)
```
> HEYHEYHEYHEYHEY!!!  
> Hey!!!
### Function with variable-length arguments (`*args`)
```python
# Define gibberish
def gibberish(*args):
    """Concatenate strings in *args together."""

    # Initialize an empty string: hodgepodge
    hodgepodge = ""

    # Concatenate the strings in args
    for word in args:
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge

# Call gibberish() with one string: one_word
one_word = gibberish('luke')

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)
```
> luke  
> lukeleiahanobidarth
### Function with variable-length keyword arguments (`**kwargs`)
* `kwargs` is a dictionary
```python
# Define report_status
def report_status(** kwargs):
    """Print out the status of a movie character."""

    print("\nBEGIN: REPORT\n")

    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)

    print("\nEND REPORT")

# First call to report_status()
report_status(name='luke', affiliation='jedi')

# Second call to report_status()
report_status(name='anakin', affiliation='sith lord', status='deceased')
```
> BEGIN: REPORT  
>   
> name: luke  
> affiliation: jedi  
>   
> END REPORT  
>   
> BEGIN: REPORT  
>   
> status: deceased  
> name: anakin  
> affiliation: sith lord
## Bring it all together
### Bringing it all together (1)
```python
# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1

        # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df)

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'source')

# Print result1 and result2
print(result1)
print(result2)
```
> {'und': 2, 'en': 97, 'et': 1}  
> {'<a href="http://twitter.com/#!/download/ipad" rel="nofollow">Twitter for iPad</a>': 6, '<a href="http://www.google.com/" rel="nofollow">Google</a>': 2, '<a href="http://linkis.com" rel="nofollow">Linkis.com</a>': 2, '<a href="http://ifttt.com" rel="nofollow">IFTTT</a>': 1, '<a href="http://www.twitter.com" rel="nofollow">Twitter for BlackBerry</a>': 2, '<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>': 33, '<a href="http://www.facebook.com/twitter" rel="nofollow">Facebook</a>': 1, '<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>': 24, '<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>': 26, '<a href="http://rutracker.org/forum/viewforum.php?f=93" rel="nofollow">newzlasz</a>': 2, '<a href="http://www.myplume.com/" rel="nofollow">Plume\xa0for\xa0Android</a>': 1}
### Bringing it all together (2)
```python
# Define count_entries()
def count_entries(df, *args):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    #Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Iterate over column names in args
    for col_name in args:
    
        # Extract column from DataFrame: col
        col = df[col_name]
    
        # Iterate over the column in DataFrame
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
    
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)
```
> {'und': 2, 'en': 97, 'et': 1}  
> {'<a href="http://twitter.com/#!/download/ipad" rel="nofollow">Twitter for iPad</a>': 6, '<a href="http://www.google.com/" rel="nofollow">Google</a>': 2, 'et': 1, '<a href="http://ifttt.com" rel="nofollow">IFTTT</a>': 1, 'und': 2, '<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>': 24, '<a href="http://www.myplume.com/" rel="nofollow">Plume\xa0for\xa0Android</a>': 1, '<a href="http://linkis.com" rel="nofollow">Linkis.com</a>': 2, 'en': 97, '<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>': 33, '<a href="http://www.facebook.com/twitter" rel="nofollow">Facebook</a>': 1, '<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>': 26, '<a href="http://rutracker.org/forum/viewforum.php?f=93" rel="nofollow">newzlasz</a>': 2, '<a href="http://www.twitter.com" rel="nofollow">Twitter for BlackBerry</a>': 2}
## Lambda functions and error-handling
### Pop quiz on lambda functions
* How would you write a lambda function `add_bangs` that adds three exclamation points `'!!!'` to the end of a string `a`?
* How would you call `add_bangs` with the argument `hello'`?
> The lambda function definition is: `add_bangs = (lambda a: a + '!!!')`, and the function call is: `add_bangs('hello')`.
### Writing a lambda function you already know
```python
# Define echo_word as a lambda function: echo_word
echo_word = (lambda word1, echo: word1*echo)

# Call echo_word: result
result = echo_word('hey', 5)

# Print result
print(result)
```
> heyheyheyheyhey
### Map() and lambda functions  
**`Map`**: applies a function to all the items in an input_list and return `print(list(map))`
```python
# Create a list of strings: spells
spells = ["protego", "accio", "expecto patronum", "legilimens"]

# Use map() to apply a lambda function over spells: shout_spells
shout_spells = map(lambda spell: spell + '!!!', spells)

# Convert shout_spells to a list: shout_spells_list
shout_spells_list = list(shout_spells)

# Convert shout_spells into a list and print it
print(shout_spells_list)
```
> ['protego!!!', 'accio!!!', 'expecto patronum!!!', 'legilimens!!!']
### Filter() and lambda functions  
* **`filter`** creates a list of elements for which a function returns true
* In the `filter()` call, pass a lambda function and the list of strings, `fellowship`. The lambda function should check if the number of characters in a string `member` is greater than 6; use the `len()` function to do this. Assign the resulting filter object to `result`.
```python
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda member: len(member) > 6, fellowship)

# Convert result to a list: result_list
result_list = list(result)

# Convert result into a list and print it
print(result_list)
```
> ['samwise', 'aragorn', 'legolas', 'boromir']
### Reduce() and lambda functions
* **`Reduce`** is a really useful function for performing some computation on a list and returning the result
* In the `reduce()` call, pass a lambda function that takes two string arguments `item1` and `item2` and concatenates them; also pass the list of strings, `stark`. Assign the result to `result`. The first argument to `reduce()` should be the lambda function and the second argument is the list `stark`
```python
# Import reduce from functools
from functools import reduce

# Create a list of strings: stark
stark = ['robb', 'sansa', 'arya', 'eddard', 'jon']

# Use reduce() to apply a lambda function over stark: result
result = reduce(lambda item1, item2: item1 + item2, stark)

# Print the result
print(result)
```
> robbsansaaryaeddardjon
### Pop quiz about errors
which function call raises an error and what type of error is raised  
> len('There is a beast in every man and it stirs when you put a sword in his hand.')  
>   
> len(['robb', 'sansa', 'arya', 'eddard', 'jon'])  
>   
> len(525600)  
>   
> len(('jaime', 'cersei', 'tywin', 'tyrion', 'joffrey'))  
> The call `len(525600)` raises a `TypeError`.
### Error handling with try-except

```python
# Define shout_echo
def shout_echo(word1, echo):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Initialize empty strings: echo_word, shout_words
    echo_word = ""
    shout_words = ""
    

    # Add exception handling with try-except
    try:
        # Concatenate echo copies of word1 using *: echo_word
        echo_word = word1 * echo

        # Concatenate '!!!' to echo_word: shout_words
        shout_words = echo_word + '!!!'
    except:
        # Print error message
        print("word1 must be a string and echo must be an integer.")

    # Return shout_words
    return shout_words

# Call shout_echo
shout_echo("particle", echo="accelerator")
```

> word1 must be a string and echo must be an integer.
### Error handling by raising an error
```python
# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Raise an error with raise
    if echo < 0:
        raise ValueError('echo must be greater than 0')

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word

# Call shout_echo
shout_echo("particle", echo=-2)
```
> ValueError: echo must be greater than 0
### Bringing it all together (1)
```python
# Select retweets from the Twitter DataFrame: result
result = filter(lambda x: 'RT' == x[0:2], tweets_df['text'])

# Create list from filter object result: res_list
res_list = list(result)

# Print all retweets in res_list
for tweet in res_list:
    print(tweet)
```
> RT @bpolitics: .@krollbondrating's Christopher Whalen says Clinton is the weakest Dem candidate in 50 years https://t.co/pLk7rvoRSn https:/…  
> RT @HeidiAlpine: @dmartosko Cruz video found.....racing from the scene.... #cruzsexscandal https://t.co/zuAPZfQDk3
### Bringing it all together (2)
```python
# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Add try block
    try:
        # Extract column from DataFrame: col
        col = df[col_name]
        
        # Iterate over the column in dataframe
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1
    
        # Return the cols_count dictionary
        return cols_count

    # Add except block
    except:
        print('The DataFrame does not have a ' + col_name + ' column.')

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Print result1
print(result1)

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang1')
```
> {'und': 2, 'en': 97, 'et': 1}  
> The DataFrame does not have a lang1 column.
### Bringing it all together (3)
```python
# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Raise a ValueError if col_name is NOT in DataFrame
    if col_name not in df.columns:
        raise ValueError('The DataFrame does not have a ' + col_name + ' column.')

    # Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1
        
        # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang1')

# Print result1
print(result1)
```
> Traceback (most recent call last):  
>   File "<stdin>", line 30, in <module>  
>     result1 = count_entries(tweets_df, 'lang1')  
>   File "<stdin>", line 8, in count_entries  
>     raise ValueError('The DataFrame does not have a ' + col_name + ' column.')  
> ValueError: The DataFrame does not have a lang1 column.