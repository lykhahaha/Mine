import sys

class Counter(dict):
    def __missing__(self, key):
        return 0

def string_compression(str_1):
    counter = Counter()
    for word in str_1:
        counter[word] += 1

    str_output = ''
    for word, count_word in counter.items():
        str_output = ''.join([str_output, str(word), str(count_word)])

    return str_output

if __name__ == '__main__':
    print(string_compression(sys.argv[-1]))