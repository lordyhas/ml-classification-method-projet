def find_best_split(size, expected = 5):
    splits = []
    for i in range(1, size+1):
        if size % i == 0:
            splits.append(i)
            if i > expected:
                return i, splits
    return splits[-1], splits
