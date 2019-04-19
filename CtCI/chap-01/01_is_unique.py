import sys

def is_unique(string):
    string_list = []
    for word in string:
        if word in string_list:
            return False
        string_list.append(word)

    return True

if __name__ == '__main__':
    print(is_unique(sys.argv[-1]))