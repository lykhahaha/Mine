import sys

class Counter(dict):
    def __missing__(self, key):
        return 0

def is_permutation(str_1, str_2):
    counter = Counter()
    for word in str_1:
        counter[word] += 1
    for word in str_2:
        if not word in counter.keys():
            return False
        counter[word] -= 1
        if counter[word] == 0:
            del counter[word]
    return len(counter) == 0

if __name__ == '__main__':
    print(is_permutation(sys.argv[-2], sys.argv[-1]))