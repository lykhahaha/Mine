from collections import Counter
import math
import os
import random
import re
import sys

def sockMerchant(n, arr):
    num_pair = 0
    for value, counter in Counter(arr).items():
        if counter >= 2:
            num_pair += counter//2
    
    return num_pair

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = sockMerchant(n, arr)

    print(result)