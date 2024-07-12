from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
nx = 1000  # Number of spatial points
nt = 500  # Number of time steps
L = 10.0  # Length of the domain
c = 1.0   # Advection speed
dx = L / nx
dt = 0.01
CFL = c * dt / dx  # CFL number

# Domain decomposition
local_nx = nx // size
start = rank * local_nx
end = start + local_nx
if rank == size - 1:
    end = nx  # Ensure the last process gets the remainder if nx is not perfectly divisible

# Initialize solution array
u = np.zeros(local_nx + 2)  # Add ghost cells
unew = np.zeros_like(u)

# Initial condition: Gaussian pulse
x = np.linspace(start * dx, (end - 1) * dx, local_nx)
u[1:-1] = np.exp(-((x - L/2)**2) / (2 * (0.1)**2))

# Time stepping loop
for t in range(nt):
    # Communicate ghost cells
    if rank > 0:
        comm.Sendrecv(u[1], dest=rank-1, sendtag=0, recvbuf=u[0], source=rank-1, recvtag=1)
    if rank < size - 1:
        comm.Sendrecv(u[-2], dest=rank+1, sendtag=1, recvbuf=u[-1], source=rank+1, recvtag=0)

    # Finite volume update
    for i in range(1, local_nx + 1):
        unew[i] = u[i] - CFL * (u[i] - u[i-1])

    # Update solution
    u[1:-1] = unew[1:-1]

# Gather results to the root process
global_u = None
if rank == 0:
    global_u = np.zeros(nx)
comm.Gather(u[1:-1], global_u, root=0)

# Output the result
if rank == 0:
    import matplotlib.pyplot as plt
    x_global = np.linspace(0, L, nx)
    plt.plot(x_global, global_u, label='t={}'.format(nt*dt))
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.show()
