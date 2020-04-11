import numpy as np

l1 = [4, 5, 6, 7, 9]
l2 = list(np.zeros(31))

p = 35
n_zone = (100 / p)
k = len(l2) / n_zone


if k % 1 > 0:
    k = int(k) + 1
else:
    k = int(k)

print(len(l2))
print(k)

l1l = [l2[i:i + k] for i in range(0, len(l2), k)]
print(l1l)
