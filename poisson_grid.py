import numpy as np
from mpi4py import MPI
from psydac.ddm.partition import compute_dims

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from numba import njit

comm = MPI.COMM_WORLD
nb_procs = comm.Get_size()
rank = comm.Get_rank()

nb_neighbours = 4
N = 0
E = 1
S = 2
W = 3

neighbour = np.zeros(nb_neighbours, dtype=np.int8)
ntx = 10
nty = 10

Nx = ntx+2
Ny = nty+2

npoints  =  [ntx, nty]
p1 = [2,2]
P1 = [False, False]
reorder = True

''' Grid spacing '''
hx = 1/(ntx+1.)
hy = 1/(nty+1.)

''' Equation Coefficients '''
coef = np.zeros(3)
coef[0] = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
coef[1] = 1./(hx*hx)
coef[2] = 1./(hy*hy)

def create_2d_cart(npoints, p1, P1, reorder):
    
    # Store input arguments                                                                                                                                                                                                                                               
    npts    = tuple(npoints)
    pads    = tuple(p1)
    periods = tuple(P1)
    reorder = reorder
    
    nprocs, block_shape = compute_dims(nb_procs, npts, pads )
    
    dims = nprocs
    
    if (rank == 0):
        print("Execution poisson with",nb_procs," MPI processes\n"
               "Size of the domain : ntx=",npoints[0], " nty=",npoints[1],"\n"
               "Dimension for the topology :",dims[0]," along x", dims[1]," along y\n"
               "-----------------------------------------")  
    
    '''
    * Creation of the Cartesian topology
    '''
    
    cart2d = comm.Create_cart(dims = dims ,periods = periods ,reorder = reorder)
    
    return dims, cart2d

def create_2dCoords(cart2d, npoints, dims):
   
    ''' Create 2d coordinates of each process '''
    
    ''' 1. Get coordinates from cart '''
    coords = cart2d.Get_coords(rank)
    
    z = [npoints[0]/dims[0],npoints[1]/dims[1]]
    
    ''' 2. Calculate sx, ex, sy and ey '''
    sx = coords[0]*z[0]+1
    ex = coords[0]*z[0]+z[0]
    
    sy = coords[1]*z[1]+1
    ey = coords[1]*z[1]+z[1]

    print("Rank in the topology :",rank," Local Grid Index :", sx, " to ",ex," along x, ",sy, " to", ey," along y")
    
    return int(sx), int(ex), int(sy), int(ey)

def create_neighbours(cart2d):

    # Get my northern and southern neighbours
    neighbour[S] , neighbour[N] = cart2d.Shift( direction = 1 , disp=1 )
    
    # Get my western and eastern neighbours
    neighbour[W] , neighbour[E] = cart2d.Shift( direction = 0 , disp=1 )

    
    print("Process", rank," neighbour: N", neighbour[N]," E",neighbour[E] ," S ",neighbour[S]," W",neighbour[W])
    
    return neighbour


'''Creation of the derived datatypes needed to exchange points with neighbours'''
def create_derived_type(sx, ex, sy, ey):
    
    '''Creation of the type_line derived datatype to exchange points with northern to southern neighbours '''


    '''Creation of the type_column derived datatype to exchange points with western to eastern neighbours '''

    '''
    We tried to create derived type but did get wanted values even for exact values of u.
    So we found that it's better to use slice instead of derived types.
    ''' 

    type_ligne = 0
    type_column = 0
    return type_ligne, type_column


''' Exchange the points at the interface '''
def communications(u, sx, ex, sy, ey, type_column, type_ligne):
    
    """ Here IDX(i, j) means index of element located at i, j in the array u """
    
    ''' Send to neighbour N and receive from neighbour S '''
    if(neighbour[N]>=0):

        sendbuf = u[IDX(sx-1,ey):IDX(ex+1,ey)+1:ey-sy+3].copy()
        comm.Send( sendbuf ,neighbour[N])
        
        recvbuf = np.zeros(ex-sx+3)
        comm.Recv( recvbuf ,neighbour[N])
        u[IDX(sx-1,sy-1):IDX(ex+1,sy-1)+1:ey-sy+3] = recvbuf
    
    
    ''' Send to neighbour S and receive from neighbour N '''
    if(neighbour[S]>=0):

        sendbuf = u[IDX(sx-1,sy):IDX(ex+1,sy)+1:ey-sy+3].copy()
        comm.Send( sendbuf ,neighbour[S])

        recvbuf = np.zeros(ex-sx+3)
        comm.Recv(recvbuf ,neighbour[S])
        u[IDX(sx-1,ey+1):IDX(ex+1,ey+1)+1:ey-sy+3] = recvbuf

        
    ''' Send to neighbour W and receive from neighbour E '''
    if(neighbour[W]>=0):

        sendbuf = u[IDX(sx,sy-1):IDX(sx,ey+1)+1].copy()
        comm.Send( sendbuf, neighbour[W])

        recvbuf = np.zeros(ey-sy+3)
        comm.Recv(recvbuf ,neighbour[W])
        u[IDX(sx-1,sy-1):IDX(sx-1,ey+1)+1] = recvbuf
    
    ''' Send to neighbour E and receive from neighbour W '''
    if(neighbour[E]>=0):

        sendbuf = u[IDX(ex,sy-1):IDX(ex,ey+1)+1].copy()
        comm.Send( sendbuf,neighbour[E])

        recvbuf = np.zeros(ey-sy+3)
        comm.Recv(recvbuf ,neighbour[E])
        u[IDX(ex+1,sy-1):IDX(ex+1,ey+1)+1] = recvbuf
       
    

'''
 * IDX(i, j) : indice de l'element i, j dans le tableau u
 * sx-1 <= i <= ex+1
 * sy-1 <= j <= ey+1
'''
def IDX(i, j):
    return int(( ((i)-(sx-1))*(ey-sy+3) + (j)-(sy-1) ))

def initialization(sx, ex, sy, ey):
    
    ''' Grid spacing in each dimension'''
    ''' Solution u and u_new at the n and n+1 iterations '''
    
    SIZE = (ex-sx+3) * (ey-sy+3)

    u       = np.zeros(int(SIZE))
    u_new   = np.zeros(int(SIZE))
    f       = np.zeros(int(SIZE))
    u_exact = np.zeros(int(SIZE))
    
    
    '''Initialition of rhs f and exact soluction '''
    for i in range(sx-1, ex+1, 1):
        x = i*hx
        for j in range(sy-1, ey+1, 1):
            y = j*hy
            u_exact[IDX(i,j)] = x*y*(x-1)*(y-1)
            f[IDX(i,j)] = 2*(x*x-x+y*y-y)

    return u, u_new, u_exact, f

def computation(u, u_new):
    
    ''' Compute the new value of u using '''
    for i in range(sx,ex+1):
        for j in range(sy,ey+1):
            u_new[IDX(i,j)] = coef[0]*(coef[1]*(u[IDX(i+1,j)]+u[IDX(i-1,j)])+coef[2]*(u[IDX(i,j+1)]+u[IDX(i,j-1)])-f[IDX(i,j)])
            
def output_results(u, u_exact):
    
    print("Exact Solution u_exact - Computed Solution u - difference")
    for itery in range(sy, ey+1, 1):
        print(u_exact[IDX(1, itery)], '-', u[IDX(1, itery)], u_exact[IDX(1, itery)]-u[IDX(1, itery)] );

        
''' Calcul for the global error (maximum of the locals errors) '''
def global_error(u, u_new):
   
    local_error = 0
     
    for iterx in range(sx, ex+1, 1):
        for itery in range(sy, ey+1, 1):
            temp = np.fabs( u[IDX(iterx, itery)] - u_new[IDX(iterx, itery)]  )
            if local_error < temp:
                local_error = temp
    
    return local_error

import meshio

def plot_2d(f, title):
    import warnings
    warnings.filterwarnings("ignore")

    f = np.reshape(f, (ex-sx+3, ey-sy+3))
    
    x = np.linspace(0, 2, ey-sy+3)
    y = np.linspace(0, 2, ex-sx+3)
    
    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = fig.gca(projection='3d')                      
    X, Y = np.meshgrid(x, y)      

    ax.plot_surface(X, Y, f, cmap=cm.viridis)
    plt.title(title)
    plt.show()

dims, cart2d   = create_2d_cart(npoints, p1, P1, reorder)
neighbour      = create_neighbours(cart2d)

sx, ex, sy, ey = create_2dCoords(cart2d, npoints, dims)

type_ligne, type_column = create_derived_type(sx, ex, sy, ey)
u, u_new, u_exact, f = initialization(sx, ex, sy, ey)


''' Time stepping '''
it = 0
convergence = False
it_max = 100000
eps = 2.e-16

''' Create progress bar '''
from tqdm.notebook import tqdm
pbar = tqdm(total=it_max)
pbar.set_description("it / max_it")

''' Elapsed time '''
t1 = MPI.Wtime()


while (not(convergence) and (it < it_max)):
    it = it+1

    temp = u.copy() 
    u = u_new.copy() 
    u_new = temp.copy()
    
    ''' Exchange of the interfaces at the n iteration '''
    communications(u, sx, ex, sy, ey, type_column, type_ligne)
   
    ''' Computation of u at the n+1 iteration '''
    computation(u, u_new)
    
    ''' Computation of the global error '''
    local_error = global_error(u, u_new)
    diffnorm = comm.allreduce(np.array(local_error), op=MPI.MAX )   
   
    ''' Stop if we obtained the machine precision '''
    convergence = (diffnorm < eps)
    
    ''' Print diffnorm for process 0 '''
    if ((rank == 0) and ((it % 10) == 0)):
        print("Iteration", it, " global_error = ", diffnorm)
    
    it += 1
    
    ''' Updating the progress bar '''
    pbar.update(1)
    
''' Elapsed time '''
t2 = MPI.Wtime()
del(pbar)


if (rank == 0):

    ''' Print convergence time for process 0 '''
    print(f'\nConvergence after {it} iterations in {t2 - t1} secs')

    ''' Compare to the exact solution on process 0 '''
    output_results(u, u_exact)

    plot_2d(u, 'Représentation des valeurs approchées')
    plot_2d(u_exact, 'Représentation des valeurs exactes')
