import random


def test_multiple_shuffles():
    original_list = [1, 2, 3, 4, 5]
    print(f"Original list: {original_list}")
    copy = original_list[:]

    # Shuffle the list multiple times and show results
    for i in range(5):  # Shuffle 5 times
        random.shuffle(copy)  # Shuffle the copied list
        print(f"Shuffle {i + 1}: {copy}")


# Test multiple shuffles on the same list
test_multiple_shuffles()
