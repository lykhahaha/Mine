import sys

def one_way(str_1, str_2):
    diff_count = abs(len(str_1) - len(str_2))
    if diff_count > 1:
        return False
    elif diff_count == 0:
        diff = 0
        for word_1, word_2 in zip(str_1, str_2):
            if word_1 != word_2:
                diff += 1
        
        if diff == 1:
            return True
        return False
    else:
        if len(str_1) > len(str_2):
            shorter, longer = str_2, str_1
        else:
            shorter, longer = str_1, str_2
        
        for i, word_shorter in enumerate(shorter):
            if word_shorter == longer[i]:
                continue
            if word_shorter != longer[i] and word_shorter == longer[i+1]:
                return True
            else:
                return False
        return True

if __name__ == '__main__':
    print(one_way(sys.argv[-2], sys.argv[-1]))