import numpy as np
from numpy import e
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

hbar = 1.0

def smooth_potential(x, v0, origin_x, scale_x, softening=0.0):
    return v0 / ((abs(x - origin_x) + softening) * scale_x)


def kinetic_energy_op(mass, delta_x, N):
    ident = np.eye(N)
    up = np.roll(ident, 1, axis=0)
    down = np.roll(ident, 1, axis=1)

    mat = up - 2. * ident + down

    return -mat * hbar / (2. * mass * delta_x**2)


# Solve for coefficients (c) for psi(t) in the **energy eigenbasis**
def solve_for_coeffs(eigenbasis, energies, psi_0):
    cj = np.linalg.inv(eigenbasis) @ psi_0
    return cj.astype(np.complex128)
    

# Phases for energy eigenbasis coefficients at time t
def phases(energies, t):
    # Calculate phases from time and energy
    return e**(-1j * energies * t / hbar)


def solve_for_psi(cj, eigenbasis, energies, t):
    return eigenbasis @ (phases(energies, t) * cj)


N = 500
v0 = 10.0
mass = 1.0
delta_x = 1.0

x = np.linspace(-10.0, 10.0, N)
V = smooth_potential(x, v0, 0.0, 10.0, softening=0.0)
psi = np.zeros_like(x, np.complex128)

KE = kinetic_energy_op(mass, delta_x, N)
H = np.diag(V) + KE

E, eigvects = np.linalg.eig(H)

print(E)

#for i in range(len(eigvects[:5])):
    #plt.plot(x, eigvects[i,:], label=f"psi_{i}(x)")
    #print(eigvects[i,:])
#plt.show()

# Design the wave function ...
desired_psi = np.zeros_like(x)
desired_psi[len(desired_psi)//3] = 1./sqrt(2.)
desired_psi[len(desired_psi)//3+1] = 1./sqrt(2.)

fig, ax = plt.subplots()

cj = solve_for_coeffs(eigvects, E, desired_psi)

def update(frame):
    psi = solve_for_psi(cj, eigvects, E, t=float(frame))
    #line_V.set_ydata(np.minimum(V/v0,1.))
    #line_psi.set_ydata(psi.real)
    line_P.set_ydata((psi * psi.conjugate()).real * 10.)


line_V, = ax.plot(x, np.maximum(np.minimum(V/abs(v0), 1.0), -1.0), label="V(x)")
#line_psi, = ax.plot(x, psi.real, label="psi(x)")
line_P, = ax.plot(x, np.ones_like(x), label="P(x)")

ax.legend()

num_frames = N * 1000000  # Adjust the number of frames as needed
animation = FuncAnimation(fig, update, frames=num_frames, interval=30)

plt.show()
#print(H)

#plt.ylim(0., 1.)
