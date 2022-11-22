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

def getAcc(pos_e, pos_i, Nx, boxsize, neff, Gmtx, Laptx, t, Vrf, w, Vdc):
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
    me = 1  # electron mass
    mi = 73000  # ion mass
    me *= neff
    mi *= neff
    pos = np.vstack((pos_e, pos_i))

    #N = pos_i.shape[0] - pos_e.shape[0]
    N = pos_i.shape[0]
    dx = boxsize / Nx
    j = np.floor(pos_e / dx).astype(int)
    jp1 = j + 1
    weight_j = (jp1 * dx - pos_e) / dx
    weight_jp1 = (pos_e - j * dx) / dx

    #jp1[jp1 == Nx] = Nx-1 # particle death
    #j[j == Nx-1] = Nx-2
    #jp1 = np.mod(jp1, Nx)  # periodic BC

    j_i = np.floor(pos_i / dx).astype(int)
    jp1_i = j_i + 1
    weight_j_i = (jp1_i * dx - pos_i) / dx
    weight_jp1_i = (pos_i - j_i * dx) / dx

    #jp1_i[jp1_i == Nx] = Nx-1 # particle death
    #j_i[j_i == Nx-1] = Nx - 2
    #jp1_i = np.mod(jp1_i, Nx)  # periodic BC

    n = np.bincount(j_i[:, 0], weights=weight_j_i[:, 0], minlength=Nx+1);
    n += np.bincount(jp1_i[:, 0], weights=weight_jp1_i[:, 0], minlength=Nx+1);
    n -= np.bincount(j[:, 0], weights=weight_j[:, 0], minlength=Nx+1);
    n -= np.bincount(jp1[:, 0], weights=weight_jp1[:, 0], minlength=Nx+1);

    n = np.delete(n, Nx)
    # zero boundary conditions for Poisson equation
    n[0] = 0
    n[Nx-1] = 0

    #n *= neff * 18080 # [V / mkm^2] = [n counts]*[e C]/[eps0 F/m]/[1E-12 mkm^2/m^2]/[1 m]
    n *= neff * 0.018080 / dx  # [V / mkm^2] = [n counts]*[e C]/[eps0 F/mkm]/[dx mkm]/[1 mkm^2]

    # Solve Poisson's Equation: laplacian(phi) = -n
    #phi_Pois_grid = spsolve(Lmtx, n - n0, permc_spec="MMD_AT_PLUS_A")
    phi_Pois_grid = spsolve(Laptx, -n, permc_spec="MMD_AT_PLUS_A")

    #Vrf *= 5.53E19  # Vrf = Vrf_volts*eps0*1E12/e
    #Vrf *= 5.53E13  # Vrf = Vrf_volts*eps0*1E6/e
    #Vdc *= 1.8E-8 # [V] = [Vdc counts]*[e C]/[eps0 F/m]//[1 m^3]
    zerros = []
    zerros = [0 for index in range(Nx)]

    # boundary conditions for Laplace equation: phi_Lap(Nx-1)/dx^2 = zerros[Nx-1]
    zerros[Nx - 1] = (Vdc - Vrf * np.sin(w * t)) / dx ** 2

    # Solve Laplace's Equation: laplacian(phi) = 0
    phi_Lap_grid = spsolve(Laptx, zerros, permc_spec="MMD_AT_PLUS_A")

    # Apply Derivative to get the Electric field
    #E_grid = - Gmtx @ phi_grid
    E_grid = - Gmtx @ (phi_Pois_grid + phi_Lap_grid)

    # Interpolate grid value onto particle locations
    #E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1] # mistake here

    # Boundary electric field
    E_grid = np.hstack((E_grid, 0))

    Ee = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]
    Ei = weight_j_i * E_grid[j_i] + weight_jp1_i * E_grid[jp1_i]

    ae = -Ee * neff / me
    ai = Ei * neff / mi

    # Unit calibration [a mkm/ ns^2] = [a V/mkm/e.m.u.] * [e C] * [1E-6 mkm/m] * [1E-12 mkm/ns^2] / [me kg/e.m.u.]
    #ae = ae * 3.18E-21
    #ai = ai * 3.18E-21
    #ae = ae * 3.18E-15
    #ai = ai * 3.18E-15
    ae = ae * 1.76E-7
    ai = ai * 1.76E-7

    return ae, ai, n


def main():
    """ Plasma PIC simulation """

    # Simulation parameters
    N = 10000000  # Number of particles. Need 200 000 000
    Nx = 20000  # Number of mesh cells
    t = 0  # current time of the simulation
    tEnd = 1000  # time at which simulation ends [ns]
    dt = 1  # timestep [1ns]
    boxsize = 2000  # periodic domain [0,boxsize] [mkm] 1000 mkm
    neff = 10  # number of real particles corresponding to count particles
    vth = 1E-3  # m/s to mkm/ns
    Te = 2.3  # electron temperature
    Ti = 0.06  # ion temperature
    sheath = 150 # initial sheath
    me = 1  # electron mass
    mi = 73000  # ion mass
    Energy_max = 5.0  # max electron energy
    deltaE = 100  # energy discretization
    Vdc0 = -10 # initial Vdc
    Vrf = 15  # RF amplitude
    w = 2 * np.pi * 0.01356  # frequency
    C = 1.4E-20  # capacity C = C0[F]/(Selectr/Smodel) Smodel = 1 mkm^2, Selectr = 7.1e10 mkm^2, C0 = 1000 pF
    plotRealTime = True  # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    np.random.seed(42)  # set the random number generator seed

    #Nh = int(N / 2)
    Te *= 1.7E12/9.1 # kT/me
    Ti *= 1.7E12/9.1/mi # kT/mi
    me *= neff
    mi *= neff
    C /= 1.6E-19 # [C] = [F/C]

    # Particle creation: position and velocity
    pos_e = np.random.rand(N, 1) * (boxsize - sheath)
    pos_i = np.random.rand(N, 1) * (boxsize - sheath)
    #pos = np.vstack((pos_e, pos_i))

    #vel_e = vth / m.sqrt(Te) * np.random.normal(0, m.sqrt(Te), size = (Nh, 1))
    #vel_i = vth / m.sqrt(Ti) * np.random.normal(0, m.sqrt(Ti), size = (N-Nh, 1))
    vel_e = vth * np.random.normal(0, m.sqrt(Te), size=(N, 1))
    vel_i = vth * np.random.normal(0, m.sqrt(Ti), size=(N, 1))
    #vel = np.vstack((vel_e, vel_i))

    # Construct matrix G to computer Gradient  (1st derivative) (BOUNDARY CONDITIONS)
    dx = boxsize / Nx
    e = np.ones(Nx)
    diags = np.array([-1, 1])
    vals = np.vstack((-e, e))
    Gmtx = sp.spdiags(vals, diags, Nx, Nx);
    Gmtx = sp.lil_matrix(Gmtx)
    #Gmtx[0, Nx - 1] = -1
    #Gmtx[Nx - 1, 0] = 1
    Gmtx[0, 0] = -2
    Gmtx[0, 1] = 2
    Gmtx[Nx - 1, Nx - 1] = 2
    Gmtx[Nx - 1, Nx - 2] = -2
    Gmtx /= (2 * dx)
    Gmtx = sp.csr_matrix(Gmtx)

    """
    # Construct matrix L to computer Laplacian (2nd derivative) for Poisson
    diags = np.array([-1, 0, 1])
    vals = np.vstack((e, -2 * e, e))
    Lmtx = sp.spdiags(vals, diags, Nx, Nx);
    Lmtx = sp.lil_matrix(Lmtx)
    Lmtx[0, Nx - 1] = 1
    Lmtx[Nx - 1, 0] = 1
    Lmtx /= dx ** 2
    Lmtx = sp.csr_matrix(Lmtx)
    """

    # Construct matrix L to computer Laplacian (2nd derivative) for Laplace (BOUNDARY CONDITIONS)
    diags = np.array([-1, 0, 1])
    #diags = np.array([0, 1, 2])
    vals = np.vstack((e, -2 * e, e))
    Laptx = sp.spdiags(vals, diags, Nx, Nx);
    Laptx = sp.lil_matrix(Laptx)
    Laptx[0, 0] = 1
    Laptx[0, 1] = 0
    #Laptx[0, Nx - 1] = 0
    #Laptx[Nx - 1, 0] = 0
    Laptx[Nx - 1, Nx - 2] = 0
    Laptx[Nx - 1, Nx - 1] = 1
    Laptx /= dx ** 2
    Laptx = sp.csr_matrix(Laptx)

    # calculate initial gravitational accelerations
    acc_e, acc_i, n = getAcc(pos_e, pos_i, Nx, boxsize, neff, Gmtx, Laptx, 0, Vrf, w, Vdc0)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    I = [0 for index in range(Nt)]
    Vdc = [0 for index in range(Nt)]
    q = 0
    Vdc[0] = Vdc0

    # prep figure
    fig = plt.figure(figsize=(6, 6), dpi=80)

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel_e += acc_e * dt / 2.0
        vel_i += acc_i * dt / 2.0

        # Concentration from coordinate
        """
        # plot in real time - color 1/2 particles blue, other half red
        if plotRealTime or (i == Nt - 1):
            plt.cla()
            plt.plot(np.multiply(dx, range(Nx)), n)

            plt.pause(0.001)
            print(acc_e)
        """
        # acceleration from coordinate
        """
        # plot in real time - color 1/2 particles blue, other half red
        if plotRealTime or (i == Nt - 1):
            plt.cla()
            #plt.plot(np.multiply(dx, range(Nx)), n)
            plt.scatter(pos_e, acc_e, s=.4, color='blue', alpha=0.5)
            plt.axis([0, boxsize, -1E-4, 1E-4])
            plt.xlabel('x')
            plt.ylabel('ae')
            plt.pause(0.001)
            print(n)
        """
        # drift
        pos_e += vel_e * dt
        pos_i += vel_i * dt

        # boundary condition and particle death
        #pos = np.mod(pos, boxsize) # periodic boundary condition

        #pos_e[pos_e >= boxsize] = boxsize - 0.5 * dx # particle death
        be_b = np.where(pos_e >= boxsize)
        I[i] = -len(be_b[0])
        pos_e = np.delete(pos_e, be_b[0], axis=0)
        vel_e = np.delete(vel_e, be_b[0], axis=0)
        acc_e = np.delete(acc_e, be_b[0], axis=0)
        be = np.where(pos_e <= 0)
        pos_e = np.delete(pos_e, be[0], axis = 0)
        vel_e = np.delete(vel_e, be[0], axis = 0)
        acc_e = np.delete(acc_e, be[0], axis = 0)

        #pos_i[pos_i >= boxsize] = boxsize - 0.5 * dx # particle death
        bi_b = np.where(pos_i >= boxsize)
        I[i] += len(bi_b[0])
        pos_i = np.delete(pos_i, bi_b[0], axis=0)
        vel_i = np.delete(vel_i, bi_b[0], axis=0)
        acc_i = np.delete(acc_i, bi_b[0], axis=0)
        bi = np.where(pos_i <= 0)
        pos_i = np.delete(pos_i, bi[0], axis = 0)
        vel_i = np.delete(vel_i, bi[0], axis=0)
        acc_i = np.delete(acc_i, bi[0], axis=0)

        # capacitor charge and capacity
        q += I[i]
        Vdc[i] = q/C
        #Vdc[i] *= 1.8E-8  # [V] = [Vdc counts/F]*[e C]
        #Vdc[i] *= 0.018080  # [V] = [Vdc counts]*[e C]/[eps0 F/mkm]/[1 m^3]

        # update time
        t += dt

        # particle generation
        #dNef = Nh * m.sqrt(3 * Te) / 4 / boxsize / m.sqrt(me)
        dNef = vth * N * m.sqrt(3 * Te) / 4 / boxsize
        dNe = int(dNef)
        #dNif = (N - Nh) * m.sqrt(3 * Ti) / 4 / boxsize / m.sqrt(mi)
        dNif = vth * N * m.sqrt(3 * Ti) / 4 / boxsize
        dNi = int(dNif)

        #dpos_e = np.zeros((dNe, 1))
        #dpos_i = np.zeros((dNi, 1))
        #dpos_e = np.random.rand(dNe, 1) * dx
        #dpos_i = np.random.rand(dNi, 1) * dx
        dpos_e = np.random.rand(dNe, 1) * 1000 # length electrons for dt
        dpos_i = np.random.rand(dNi, 1) * 0.7 # length ions for dt
        pos_e = np.vstack((pos_e, dpos_e))
        pos_i = np.vstack((pos_i, dpos_i))

        #dvel_e = vth / m.sqrt(Te) * np.random.normal(0, m.sqrt(Te), size=(dNe, 1))
        #dvel_i = vth / m.sqrt(Ti) * np.random.normal(0, m.sqrt(Ti), size=(dNi, 1))
        dvel_e = vth * np.random.normal(0, m.sqrt(Te), size=(dNe, 1))
        dvel_i = vth * np.random.normal(0, m.sqrt(Ti), size=(dNi, 1))
        vel_e = np.vstack((vel_e, dvel_e))
        vel_i = np.vstack((vel_i, dvel_i))

        # update accelerations
        acc_e, acc_i, n = getAcc(pos_e, pos_i, Nx, boxsize, neff, Gmtx, Laptx, t, Vrf, w, Vdc[i])

        # (1/2) kick
        vel_e += acc_e * dt / 2.0
        vel_i += acc_i * dt / 2.0

        #I[i] *= 1.6E-19
        #Vdc[i] *= 1.6E-19
        """
        #Phase diagram
        
        # plot in real time - color 1/2 particles blue, other half red
        if plotRealTime or (i == Nt - 1):
            plt.cla()
            plt.scatter(pos_e, vel_e, s=.4, color='blue', alpha=0.5)
            plt.scatter(pos_i, vel_i, s=.4, color='red', alpha=0.5)
            plt.axis([0, boxsize, -1000, 1000])

            plt.pause(0.001)

    # Save figure
    plt.xlabel('x')
    plt.ylabel('v')
    #plt.savefig('pic.png', dpi=240)
    plt.show()
    """

    """
        # Concentration from coordinate

        # plot in real time - color 1/2 particles blue, other half red
        if plotRealTime or (i == Nt - 1):
            plt.cla()
            plt.plot(np.multiply(dx, range(Nx)), n)

            plt.pause(0.001)
            print(acc_e)
    """
    """
    # Save figure
    plt.xlabel('x')
    plt.ylabel('n')
    #plt.savefig('pic.png', dpi=240)
    plt.show()
    """

    """
    # Electron energy distribution function
    energy = vel_i ** 2 / 2.0
    iedf = []
    iedf = [0 for index in range(deltaE)]
    dE = Energy_max / deltaE

    for ind in range(pos_i.shape[0]):
        if (pos_i[ind] >= boxsize - 3*dx) and (vel_i[ind] > 0):
            k = int(energy[ind] // dE)
            if k < deltaE:
                iedf[k] += 1


    plt.plot(np.multiply(dE, range(deltaE)), iedf)
    
    # Save figure
    plt.xlabel('E')
    plt.ylabel('iedf')
    #plt.savefig('iedf.png', dpi=240)
    plt.show()
    """

    plt.plot(np.multiply(dt, range(Nt)), Vdc)

    # Save figure
    plt.xlabel('t')
    plt.ylabel('Vdc')
    #plt.savefig('Vdc-t_Vrf5.png', dpi=240)
    plt.show()

    plt.plot(np.multiply(dt, range(Nt)), I)

    # Save figure
    plt.xlabel('t')
    plt.ylabel('I')
    # plt.savefig('Vdc-t_Vrf5.png', dpi=240)
    plt.show()

    plt.plot(np.multiply(dx, range(Nx)), n * dx ** 2)

    plt.show()

    return 0


if __name__ == "__main__":
    main()
