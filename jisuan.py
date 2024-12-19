import numpy as np

A = np.array([[0, 1], [1, 1]])

B = np.array([[(1+np.sqrt(5))/2, 0], [0, (1-np.sqrt(5))/2]])

C = np.array([[2, 2], [1+np.sqrt(5), 1-np.sqrt(5)]])

D = np.linalg.inv(C)

result = C@B@B@B@D

print(A@A@A)
print(result)