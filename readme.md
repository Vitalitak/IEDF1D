1D Self-consistent Particle-In-Cell (PIC) simulation model of Ion Energy Distribution Function (IEDF)

Self-consistent calculation of electric potential and electric field using Poisson and Laplace equations and calculation of charge density.

Quasi-neutral plasma with concentration of N particles per volume equals to boxsize x 1 x 1 mkm^3 at 0 coordinate. Electrode connected to capacitor with C capacity at boxsize coordinate.

Units:
[x] = mkm
[t] = ns

vel_e ~ 1100 mkm/ns

vel_i ~ 0.68 mkm/ns

dx = 0.01 mkm
dt = 0.01 ns


Current problem:
Disturbed of continuity equation. Particles generated in plasma break electric field.

N = 1000000

dNe+ = 5700
dNe-el = 2600
dNe-pl = 3000


TODO:

Electric field boundary condition at x=0

mirror reflection at x=0
