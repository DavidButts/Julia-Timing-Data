import time
import numpy as np
from numba import njit, prange

dim = 30
x_min = -1.8
x_max = 1.8
y_min = -1.8j
y_max = 1.8
@njit(parallel = True)
def julia(c,z):
	it = 0
	max_iter = 100
	while(it < max_iter):
		for y in prange(dim):
			for x in prange(dim):
				if abs(z[y][x]) < 10:
					z[y][x] = z[y][x]**2 + c
		it += 1
	return z

counts = []
data = []
for run in range(100):
	z = np.zeros((dim,dim),dtype = 'complex128')
	for l in range(dim):
		z[l] = np.linspace(x_min,x_max,dim) - np.linspace(y_min,y_max,dim)[l]
	start = time.perf_counter()
	julia((-.4+.6j),z)
	end = time.perf_counter()
	counts.append(end-start)

data.append(min(counts))
data.append(sum(counts)/len(counts))
print('minimum = ', data[0], 'average = ', data[1])
#np.savetxt(fname='Run' + str(dim) + 'min-ave', X = data)
#np.savetxt(fname='Run' + str(dim), X = counts)
