import math


def set_entropy(values):
    score = 0
    for v in values:
        if v == 0:
            score -= 0
        else:
            score -= (v * math.log2(v))

    return score


def field_entropy(values):
    score = 0
    i = 0
    while i < len(values):
        score += (values[i] * values[i + 1])
        i += 2

    return score


print("comedy", set_entropy([1 / 4, 3 / 4]))
print("drama", set_entropy([2 / 3, 1 / 3]))
print("Star Actors", set_entropy([1 / 6, 5 / 6]))
print("profit", set_entropy([4 / 7, 3 / 7]))
print("Income", field_entropy([4 / 7, set_entropy([1 / 4, 3 / 4]),
                               3 / 7, set_entropy([2 / 3, 1 / 3])]))

