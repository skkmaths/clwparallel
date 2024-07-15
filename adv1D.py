from mpi4py import MPI
import numpy as np

# Domain size (global parameters, same for all processes)
L = 1.0
Nx = 100  # Total number of cells
dx = L / Nx  # Cell size
dt = 0.01  # Time step
T = 1.0  # Final time
c = 1.0  # Advection speed

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Determine local subdomain for each process
local_Nx = Nx // size
local_start = rank * local_Nx
local_end = local_start + local_Nx if rank != size - 1 else Nx
local_x = np.linspace(local_start * dx, local_end * dx, local_Nx, endpoint=False)

# Initial condition (example: rectangular pulse)
u = np.zeros(local_Nx)
u = np.sin( 2.0* np.pi * local_x)
#u[1:-1]  = np.exp(-(local_x[1:-1] - 1.0)**2 / 0.1**2) 
#u[int(local_Nx/4):int(local_Nx/2)] = 1

# Helper function to apply periodic boundary conditions
def apply_periodic_boundary_conditions(u):
    u_left = np.roll(u, 1)
    u_right = np.roll(u, -1)
    return u_left, u_right

# Time-stepping loop
time = 0.0
while time < T:
    u_left, u_right = apply_periodic_boundary_conditions(u)

    # Send and receive boundary data with neighboring processes
    if rank > 0:
        comm.send(u[0], dest=rank-1)
        u_left[0] = comm.recv(source=rank-1)

    if rank < size - 1:
        comm.send(u[-1], dest=rank+1)
        u_right[-1] = comm.recv(source=rank+1)

    # Update solution using finite volume method
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] - (c * dt / dx) * (u[1:-1] - u_left[1:-1])

    # Update the solution array
    u = u_new

    # Update time
    time += dt

# Gather results to root process
u_global = None
if rank == 0:
    u_global = np.zeros(Nx)
comm.Gather(u, u_global, root=0)

# Plot results if rank is 0
if rank == 0:
    import matplotlib.pyplot as plt
    x = np.linspace(0, L, Nx, endpoint=False)
    plt.plot(x, u_global, label=f't={time:.2f}')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.show()
