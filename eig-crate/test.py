import sys
sys.path.insert(0, './target/debug')
import numpy as np

w = 30
mat = np.random.rand(w, w)
mat = mat @ mat.transpose()

import libeig_crate

eigvals, eigvects = libeig_crate.eigdecomp(mat)
#print(eigvals)
#print(eigvects)

print(mat @ eigvects[:,0] - eigvals[0] * eigvects[:,0])


eigvals, eigvects = np.linalg.eig(mat)
#print(eigvals.real)
#print(eigvects.real)

print(mat @ eigvects[:,0] - eigvals[0] * eigvects[:,0])

