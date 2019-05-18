import re
def countingValleys(n, s):
    det = 0
    c = 0
    is_down = False
    for char in s:
        det = det + 1 if char == 'U' else det - 1
        if det < 0:
            is_down = True
        if is_down and char == 'U':
            c += 1
            is_down = False
    return c

if __name__ == '__main__':
    n = int(input())

    s = input()

    result = countingValleys(n, s)

    print(result)