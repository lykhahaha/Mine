def binary_search(arr, target):
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high)//2
        guess = arr[mid]
        if guess == target:
            return mid
        elif guess > target:
            high = mid - 1
        else:
            low = mid + 1
    return None

arr = [1, 3, 5, 7, 9]
print(binary_search(arr, 3))
print(binary_search(arr, -1))