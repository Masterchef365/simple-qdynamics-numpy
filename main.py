import numpy as np
from matplotlib import pyplot as plt

hbar = 1.0

def smooth_potential(x, v0, origin_x, scale_x, softening=0.0):
    return v0 / ((abs(x - origin_x) + softening) * scale_x)


def kinetic_energy_op(mass, delta_x, N):
    ident = np.eye(N)
    up = np.roll(ident, 1, axis=0)
    down = np.roll(ident, 1, axis=1)

    mat = up - 2. * ident + down

    return -mat * hbar / (2. * mass * delta_x**2)


N = 5
v0 = 1.0
mass = 1.0
delta_x = 1.0

x = np.linspace(-10.0, 10.0, N)
V = smooth_potential(x, v0, 0.0, 1.0, softening=2.0)

KE = kinetic_energy_op(mass, delta_x, N)
H = KE + np.diag(V)

E, eigvects = np.linalg.eig(H)

print(E)
for i in range(len(eigvects)):
    print(eigvects[i,:])

#print(H)

#plt.ylim(0., 1.)
plt.plot(x, V, label="V(x)")
plt.legend()
plt.show()
