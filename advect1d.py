from mpi4py import MPI
import numpy as np

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Grid parameters
nx_global = 100  # Global number of cells
dx = 1.0         # Cell size

# Domain decomposition
local_nx = nx_global // size
start = rank * local_nx
end = start + local_nx - 1

# Physical parameters
velocity = 1.0
dt = 0.01
num_steps = 100

# Initialize arrays
phi = np.zeros(local_nx + 2)  # Local solution including ghost cells
phi_new = np.zeros(local_nx)  # Buffer for updated values

# Initial condition
x = np.linspace(start * dx - 0.5*dx, (end+1) * dx - 0.5*dx, local_nx + 2)
phi[1:-1] = np.exp(-(x[1:-1] - 1.0)**2 / 0.1**2)  # Example initial condition

# Time integration
for step in range(num_steps):
    # Communicate ghost cells
    if rank > 0:
        comm.Send([phi[1], MPI.DOUBLE], dest=rank-1, tag=rank)
        comm.Recv([phi[0], MPI.DOUBLE], source=rank-1, tag=rank-1)
    if rank < size - 1:
        comm.Send([phi[-2], MPI.DOUBLE], dest=rank+1, tag=rank)
        comm.Recv([phi[-1], MPI.DOUBLE], source=rank+1, tag=rank+1)

    # Compute advection using upwind scheme
    for i in range(1, local_nx + 1):
        if velocity >= 0:
            flux = velocity * (phi[i] - phi[i-1]) / dx
        else:
            flux = velocity * (phi[i+1] - phi[i]) / dx
        phi_new[i-1] = phi[i] - dt * flux

    # Update phi
    phi[1:-1] = phi_new[:]
'''
# Gather results from all processes to root (rank 0)
if rank == 0:
    full_phi = np.empty(nx_global)
else:
    full_phi = None

comm.Gather([phi[1:-1], MPI.DOUBLE], [full_phi, MPI.DOUBLE], root=0)

# Plot or analyze results (only rank 0 does this)
if rank == 0:
    import matplotlib.pyplot as plt

    plt.plot(np.linspace(0, nx_global-1, nx_global) * dx, full_phi)
    plt.xlabel('x')
    plt.ylabel('phi')
    plt.title('Advection Equation Solution')
    plt.show()
'''