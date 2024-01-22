import sys
sys.path.insert(0, './target/debug')
import numpy as np
from time import time

w = 300
mat = np.random.rand(w, w)
mat = mat @ mat.transpose()

import libeig_crate

start = time()
eigvals, eigvects = libeig_crate.eigdecomp(mat)
rust_time = time() - start
print(mat @ eigvects[:,0] - eigvals[0] * eigvects[:,0])

#print(eigvals)
#print(eigvects)



start = time()
eigvals, eigvects = np.linalg.eig(mat)
py_time = time() - start
#print(eigvals.real)
#print(eigvects.real)

print(mat @ eigvects[:,0] - eigvals[0] * eigvects[:,0])

print("Rust: ", rust_time)
print("Python: ", py_time)
