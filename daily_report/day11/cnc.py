
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 
V = np.array([-1, -2, -3])


res = []
for i in range(len(x)-len(V)+1):
    res.append( sum( V*x[i:i+len(V)] ) )

print(res)
