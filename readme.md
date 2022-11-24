1D Self-consistent Particle-In-Cell (PIC) simulation model of Ion Energy Distribution Function (IEDF)

Self-consistent calculation of electric potential and electric field using Poisson and Laplace equations and calculation of charge density.

Quasi-neutral plasma with concentration of N particles per volume equals to boxsize x 1 x 1 mkm^3 at 0 coordinate. Electrode connected to capacitor with C capacity at boxsize coordinate.

Units:
[x] = mkm
[t] = ns
vel_e ~ 1100 mkm/ns
vel_i ~ 0.68 mkm/ns
dt = 0.01 ns
dx = 0.01 mkm



Current problem:
Disturbed of continuity equation. Particles generated in plasma break electric field.