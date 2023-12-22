import numpy as np


def ohe_m(text):
    tokens = set(text.lower().split())
    length = len(tokens)
    index_map = {x: index for x, index in zip(tokens, range(length))}
    ohe_matrix = []

    for token in tokens:
        ohe = np.zeros(length)
        ohe[index_map[token]] = 1
        print(token, ohe)
        ohe_matrix.append(ohe)


ohe_m("He is good boy but he is naughty")
