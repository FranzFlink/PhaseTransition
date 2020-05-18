import numpy as np
import random as r
import os 
import time
from numba import njit
r.seed(2)


@njit


def init_grid(N):
    grid = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            grid[i][j] = np.random.choice(np.array([-1,1]))
    return grid 

@njit

def E(grid):
    E = 0
    N = len(grid)
    for i in range(N):
        for j in range(N):
           E +=  grid[i][j] * ( grid[i][(j+1) % N] + grid[i][(j-1) % N] + grid[(i+1) % N][j] + grid[(i-1) % N][j] ) 

    return -E / 2
@njit

def M(grid):
    return np.sum(grid) 


@njit

def wolff_update(grid,row,col,beta,wolff_list):

    N = len(grid)
    spin = grid[row][col]
    wolff_list.append([row,col])
    print(type(wolff_list))
    for item in wolff_list:
        row = item[0]
        col = item[1]
        for k in [-1,1]:
            if spin == grid[row][(col+k+N)%N] and np.random.random() < 1-np.exp(-2*beta):
                if ([row,(col+k+N)%N]) not in wolff_list:
                    wolff_list.append([row,(col+k+N)%N])
            if spin == grid[(row+k+N)%N][col] and np.random.random() < 1-np.exp(-2*beta):
                if ([(row+k+N)%N,col]) not in wolff_list:
                    wolff_list.append([(row+k+N)%N,col])
    return wolff_list

@njit

def wolff(beta,grid):
    N = len(grid)
    i = np.random.randint(0, N-1)
    j = np.random.randint(0, N-1)
    cluster_list = [np.int64(x) for x in range(0)]
    new_q = - grid[i][j]
    cluster_list = wolff_update(grid,i,j,beta,cluster_list)

    for elem in cluster_list:
        grid[elem[0]][elem[1]] = new_q
    k = len(cluster_list)
    cluster_list = [np.int64(x) for x in range(0)]
    
    return grid,k

@njit

def simulation(N,MCS,beta): 
    grid = init_grid(N)
    energies = np.zeros(MCS)
    magnetisations = np.zeros(MCS)
    size = np.zeros(MCS)
    for i in range(1):
        grid,k = wolff(beta, grid)    
    for i in range(MCS):
        grid,k = wolff(beta, grid)
        size[i] = k 
        energies[i] = E(grid)
        magnetisations[i] = M(grid)
    return energies, magnetisations, grid, size



betas = np.linspace(0.3,0.6,10)

N = 8
s = 0
    
Es = np.zeros(len(betas))
Ms = np.zeros(len(betas))
Sizes = np.zeros(len(betas))
Cs = np.zeros(len(betas))
Chis = np.zeros(len(betas))


t_0 = time.time()
for beta in betas :
    
    
    print(s)
    en, ma, grid, size = simulation(N,10000,beta)
    Es[s] = np.mean(en) / N**2
    Sizes[s] = np.mean(abs(size)) / N**2
    Cs[s] = beta**2 * (np.mean(en**2)-np.mean(en)**2) / N**2
    Ms[s] = np.mean(ma) / N**2
    Chis[s] = beta * np.mean(ma**2) * N**2
    
    
    
    s += 1
np.savetxt('2DMetropolis_state_-1_1__N'+str(N),(betas,Es,Cs,Ms,Chis,Sizes),header = "beta    <e>    <C>     <m>     <Chi>     <Size>   ")  
t_1 = time.time()
print(t_1-t_0)
 