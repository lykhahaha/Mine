# Find the smallest value in an array
def find_min(arr):
    # get first value as initial min value
    min_value = arr[0]
    # store index of smallest value
    min_index = 0
    for i in range(len(arr)):
        if arr[i] < min_value:
            min_value = arr[i]
            min_index = i
    
    return min_index

def selection_sort(arr):
    sorted_arr = []
    for i in range(len(arr)):
        # Find the smallest element in the array and adds it to the new array
        min_value = find_min(arr)
        sorted_arr.append(arr.pop(min_value))
    return sorted_arr

print(selection_sort([5, 3, 6, 2, 10]))