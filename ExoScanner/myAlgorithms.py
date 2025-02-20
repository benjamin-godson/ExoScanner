# This file implements some standard-algorithms which are used all over the
# program. Currently there is only a function which calculates the rolling
# average over data.

def rolling(numbers, window_size):
    i = 0
    moving_averages = []
    while i < len(numbers) - window_size:
        this_window = numbers[i : i + window_size]

        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1

    return moving_averages