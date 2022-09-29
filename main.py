import math as m
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

"""
Create Your Own Plasma PIC Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the 1D Two-Stream Instability
Code calculates the motions of electron under the Poisson-Maxwell equation
using the Particle-In-Cell (PIC) method

"""


def getAcc(pos, Nx, Nh, boxsize, n0, Gmtx, Lmtx, Laptx, t, Vrf, w):
    """
    Calculate the acceleration on each particle due to electric field
    pos      is an Nx1 matrix of particle positions
    Nx       is the number of mesh cells
    Nh       is number of electrons
    boxsize  is the domain [0,boxsize]
    n0       is the electron number density
    Gmtx     is an Nx x Nx matrix for calculating the gradient on the grid
    Lmtx     is an Nx x Nx matrix for calculating the laplacian on the grid
    a        is an Nx1 matrix of accelerations
    t        is an current time
    Vrf      is an RF amplitude
    """
    # Calculate Electron Number Density on the Mesh by
    # placing particles into the 2 nearest bins (j & j+1, with proper weights)
    # and normalizing

    """
    Charge density, charge mobility
    """

    N = pos.shape[0]
    dx = boxsize / Nx
    j = np.floor(pos[0:Nh] / dx).astype(int)
    jp1 = j + 1
    weight_j = (jp1 * dx - pos[0:Nh]) / dx
    weight_jp1 = (pos[0:Nh] - j * dx) / dx
    jp1 = np.mod(jp1, Nx)  # periodic BC

    n = np.bincount(j[:, 0], weights=weight_j[:, 0], minlength=Nx);

    ne_boxsize = np.bincount(jp1[:, 0], weights=weight_jp1[:, 0], minlength=Nx);
    n += ne_boxsize[0:Nx]

    j_i = np.floor(pos[Nh:] / dx).astype(int)
    jp1_i = j_i + 1
    weight_j_i = (jp1_i * dx - pos[Nh:]) / dx
    weight_jp1_i = (pos[Nh:] - j_i * dx) / dx
    jp1_i = np.mod(jp1_i, Nx)  # periodic BC

    n -= np.bincount(j_i[:, 0], weights=weight_j_i[:, 0], minlength=Nx);

    ni_boxsize = np.bincount(jp1_i[:, 0], weights=weight_jp1_i[:, 0], minlength=Nx);
    n -= ni_boxsize[0:Nx]

    n *= n0 * boxsize / N / dx

    # Solve Poisson's Equation: laplacian(phi) = n-n0
    #phi_Pois_grid = spsolve(Lmtx, n - n0, permc_spec="MMD_AT_PLUS_A")
    phi_Pois_grid = spsolve(Laptx, n, permc_spec="MMD_AT_PLUS_A")

    zerros = []
    zerros = [0 for index in range(Nx)]
    zerros[Nx-1] = n[Nx-1] - Vrf * np.sin(w*t)

    # Solve Laplace's Equation: laplacian(phi) = 0
    phi_Lap_grid = spsolve(Laptx, zerros, permc_spec="MMD_AT_PLUS_A")

    # Apply Derivative to get the Electric field
    #E_grid = - Gmtx @ phi_grid
    E_grid = - Gmtx @ (phi_Pois_grid + phi_Lap_grid)

    # Interpolate grid value onto particle locations
    E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]

    a = -E

    return a


def main():
    """ Plasma PIC simulation """

    # Simulation parameters
    N = 100000  # Number of particles
    Nx = 1000  # Number of mesh cells
    t = 0  # current time of the simulation
    tEnd = 50  # time at which simulation ends
    dt = 1  # timestep
    boxsize = 100  # periodic domain [0,boxsize]
    n0 = 1  # electron number density
    #vb = 3  # beam velocity
    vb = 0  # beam velocity
    vth = 1  # beam width
    #A = 0.1  # perturbation
    Te = 2.3  # electron temperature
    Ti = 0.06  # ion temperature
    me = 1 # electron mass
    mi = 73000 # ion mass
    Energy_max = 5.0  # max electron energy
    deltaE = 100  # energy discretization
    Vrf = 15  # RF amplitude
    w = 2 * np.pi * 13560000  # frequency
    plotRealTime = True  # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    np.random.seed(42)  # set the random number generator seed

    pos = np.random.rand(N, 1) * boxsize
    """
    vel = vth * np.random.randn(N, 1) + vb
    Nh = int(N / 2)
    vel[Nh:] *= -1
    """
    Nh = int(N / 2)

    vel_el = vth / m.sqrt(Te) * np.random.normal(0, m.sqrt(Te), size = (Nh, 1))
    vel_ions = vth / m.sqrt(Ti) * np.random.normal(0, m.sqrt(Ti), size = (N-Nh, 1))

    #vel_el = vth * np.random.normal(0, m.sqrt(Te), size=(Nh, 1))
    #vel_ions = vth * np.random.normal(0, m.sqrt(Ti), size=(N - Nh, 1))

    vel = np.vstack((vel_el, vel_ions))

    # add perturbation
    #vel *= (1 + A * np.sin(2 * np.pi * pos / boxsize))

    # Construct matrix G to computer Gradient  (1st derivative)
    dx = boxsize / Nx
    e = np.ones(Nx)
    diags = np.array([-1, 1])
    vals = np.vstack((-e, e))
    Gmtx = sp.spdiags(vals, diags, Nx, Nx);
    Gmtx = sp.lil_matrix(Gmtx)
    Gmtx[0, Nx - 1] = -1
    Gmtx[Nx - 1, 0] = 1
    Gmtx /= (2 * dx)
    Gmtx = sp.csr_matrix(Gmtx)

    # Construct matrix L to computer Laplacian (2nd derivative) for Poisson
    diags = np.array([-1, 0, 1])
    vals = np.vstack((e, -2 * e, e))
    Lmtx = sp.spdiags(vals, diags, Nx, Nx);
    Lmtx = sp.lil_matrix(Lmtx)
    Lmtx[0, Nx - 1] = 1
    Lmtx[Nx - 1, 0] = 1
    Lmtx /= dx ** 2
    Lmtx = sp.csr_matrix(Lmtx)

    # Construct matrix L to computer Laplacian (2nd derivative) for Laplace
    #diags = np.array([-1, 0, 1])
    diags = np.array([0, 1, 2])
    vals = np.vstack((e, -2 * e, e))
    Laptx = sp.spdiags(vals, diags, Nx, Nx);
    Laptx = sp.lil_matrix(Laptx)
    #Laptx[0, Nx - 1] = 0
    #Laptx[Nx - 1, 0] = 0
    #Laptx[Nx - 1, Nx - 2] = 0
    #Laptx[Nx - 1, Nx - 1] = 1
    Laptx /= dx ** 2
    Laptx = sp.csr_matrix(Laptx)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, Nx, Nh, boxsize, n0, Gmtx, Lmtx, Laptx, 0, Vrf, w)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # prep figure
    fig = plt.figure(figsize=(5, 4), dpi=80)

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        #vel += acc * dt / 2.0
        for ind in range(N):
            if (pos[ind] <= boxsize - dx) and (ind < Nh):
                vel[ind] += acc[int(pos[ind]//dx), 0] * dt / 2.0 / me
            elif (pos[ind] <= boxsize - dx) and (ind >= Nh):
                vel[ind] -= acc[int(pos[ind]//dx), 0] * dt / 2.0 / mi
            else:
                vel[ind] = 0


        # drift (and apply periodic boundary conditions)
        pos += vel * dt
        pos = np.mod(pos, boxsize)

        # update time
        t += dt

        # update accelerations
        acc = getAcc(pos, Nx, Nh, boxsize, n0, Gmtx, Lmtx, Laptx, t, Vrf, w)

        # (1/2) kick
        #vel += acc * dt / 2.0
        #vel[0:Nh] += acc * dt / 2.0 / me
        #vel[Nh:] -= acc * dt / 2.0 / mi
        for ind in range(N):
            if (pos[ind] <= boxsize - dx) and (ind < Nh):
                vel[ind] += acc[int(pos[ind]//dx), 0] * dt / 2.0 / me
            elif (pos[ind] <= boxsize - dx) and (ind >= Nh):
                vel[ind] -= acc[int(pos[ind]//dx), 0] * dt / 2.0 / mi
            else:
                vel[ind] = 0


        """
        Phase diagram
        """
        # plot in real time - color 1/2 particles blue, other half red
        if plotRealTime or (i == Nt - 1):
            plt.cla()
            plt.scatter(pos[0:Nh], vel[0:Nh], s=.4, color='blue', alpha=0.5)
            plt.scatter(pos[Nh:], vel[Nh:], s=.4, color='red', alpha=0.5)
            plt.axis([0, boxsize, -10, 10])

            plt.pause(0.001)

    # Save figure
    plt.xlabel('x')
    plt.ylabel('v')
    plt.savefig('pic.png', dpi=240)
    plt.show()

    """
    # Electron energy distribution function
    energy = vel ** 2 / 2.0
    iedf = []
    iedf = [0 for index in range(deltaE)]
    dE = Energy_max / deltaE

    for ind in range(Nh, N):
        if (pos[ind] >= boxsize - 100 * dx) and (vel[ind] > 0):
            k = int(energy[ind] // dE)
            if k < deltaE:
                iedf[k] += 1


    plt.plot(np.multiply(dE, range(deltaE)), iedf)
    
    # Save figure
    plt.xlabel('E')
    plt.ylabel('iedf')
    plt.savefig('pic.png', dpi=240)
    plt.show()

    """


    return 0


if __name__ == "__main__":
    main()
