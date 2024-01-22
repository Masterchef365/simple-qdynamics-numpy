import sys
sys.path.insert(0, './target/debug')
import numpy as np

w = 10
mat = np.random.rand(w, w)

import libeig_crate
eigvals, eigvects = libeig_crate.eigdecomp(mat)
print(eigvals)
