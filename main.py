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
v0 = -10.0
mass = 1.0
delta_x = 1.0

x = np.linspace(-10.0, 10.0, N)
V = smooth_potential(x, v0, 0.0, 1.0, softening=0.1)
V += smooth_potential(x, v0, 8.0, 1.0, softening=0.1)
V /= 2.0
psi = np.zeros_like(x, np.complex128)

KE = kinetic_energy_op(mass, delta_x, N)
H = np.diag(V) + KE

# Box instead of ring
#H[0,-1] = 0.0
#H[-1,0] = 0.0

E, eigvects = np.linalg.eig(H)

# Sort energies and corresponding eigenvectors
idx = E.argsort()   
E = E[idx]
eigvects = eigvects[:,idx]

#print("Energy eigenvalues: ", E)

# Design the initial wave function ...
desired_init_psi = np.zeros_like(x)
desired_init_psi[N//2 + 20] = 1.0

fig, ax = plt.subplots()

cj = solve_for_coeffs(eigvects, E, desired_init_psi)

psi_init_energy = np.sum((cj * cj.conjugate()).real * E)
print("Psi init energy: ", psi_init_energy)

P_display_mult = 20.0

#most_similar = np.argmin(np.abs(E - psi_init_energy))
#cj = np.zeros_like(cj)
#cj[most_similar] = 1.0

#print("Coefficients: ", cj)

#print(cj)

#cj = np.zeros_like(cj)
#cj[0] = 1.0
#cj[0] = 1.0/sqrt(2.0)
#cj[1] = 1.0/sqrt(2.0)
#cj[len(cj)//3] = 1.0
#cj[len(cj)//3+1] = 1.0

print("Computing average...")
n_avg = 100
avg = np.zeros_like(x)
step = 3.0
for i in range(n_avg):
    psi = solve_for_psi(cj, eigvects, E, t=float(i)*step)
    P = (psi * psi.conjugate()).real
    avg += P
avg /= float(n_avg)

def update(frame):
    psi = solve_for_psi(cj, eigvects, E, t=float(frame))
    #line_V.set_ydata(np.minimum(V/v0,1.))
    P = (psi * psi.conjugate()).real

    line_P.set_ydata(P * P_display_mult)


    #line_psi.set_ydata(psi.real)
    #plt.savefig(f"anim/{frame:04}.png")


line_V, = ax.plot(x, np.ones_like(x)*psi_init_energy, label="<H>")
line_V, = ax.plot(x, V, label="V(x)")
line_P, = ax.plot(x, np.ones_like(x), label=f"P(x) (scaled by {P_display_mult}x)")
#line_psi, = ax.plot(x, desired_init_psi, label="init psi")

avg_analytical = eigvects**2 @ cj**2
#avg = (cj.conjugate() * cj)
line_avg_P, = ax.plot(x, avg_analytical.real * P_display_mult, label=f"Average (analytical)")

line_avg_P, = ax.plot(x, avg.real * P_display_mult, label=f"Average (empirical)")

#line_cj, = ax.plot(x, cj, label=f"Coefficients")

ax.legend()

num_frames = N * 10000  # Adjust the number of frames as needed
animation = FuncAnimation(fig, update, frames=num_frames, interval=30)

plt.show()
