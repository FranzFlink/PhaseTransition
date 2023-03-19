import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from multiprocessing import Pool

def site_percolation(L, p):
    """
    Site percolation in a square lattice of size L x L with probability p.
    Returns the percolation map and the number of clusters.
    """
    # Generate a lattice with random site occupation probabilities
    
    rand_latt = np.random.rand(L, L, L)
    lattice = rand_latt < p

    
    # Initialize the label array
    labels = np.zeros((L, L, L), dtype=int)

    # Assign labels to the clusters
    label = 1
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if lattice[i, j, k]:
                    neighbors = []
                    if i > 0 and lattice[i-1, j, k]:
                        neighbors.append(labels[i-1, j, k])
                    if j > 0 and lattice[i, j-1, k]:
                        neighbors.append(labels[i, j-1, k])
                    if k > 0 and lattice[i, j, k-1]:
                        neighbors.append(labels[i, j, k-1])
                    if not neighbors:
                        labels[i, j, k] = label
                        label += 1
                    else:
                        neighbors = np.unique(neighbors)
                        labels[i, j, k] = neighbors[0]
                        for neighbor in neighbors[1:]:
                            labels[labels == neighbor] = labels[i, j, k]

    return labels, np.array(lattice, dtype=int)

def site_percolation(L, p):
    """
    Site percolation in a square lattice of size L x L with probability p.
    Returns the percolation map and the number of clusters.
    """
    # Generate a lattice with random site occupation probabilities
    
    rand_latt = np.random.rand(L, L)
    lattice = rand_latt < p

    
    # Initialize the label array
    labels = np.zeros((L, L), dtype=int)

    # Assign labels to the clusters
    label = 1
    for i in range(L):
        for j in range(L):
            if lattice[i, j]:
                neighbors = []
                if i > 0 and lattice[i-1, j]:
                    neighbors.append(labels[i-1, j])
                if j > 0 and lattice[i, j-1]:
                    neighbors.append(labels[i, j-1])
                if not neighbors:
                    labels[i, j] = label
                    label += 1
                else:
                    neighbors = np.unique(neighbors)
                    labels[i, j] = neighbors[0]
                    for neighbor in neighbors[1:]:
                        labels[labels == neighbor] = labels[i, j]

    return labels, np.array(lattice, dtype=int)

def measure_system_2D(L, p):
    """
    Measures the basic quantities of the 2D site percolation system with lattice size L
    for site occupation probabilities in the range p_range.
    Returns the number of clusters, the largest cluster size, and the percolation probability
    for each probability in p_range.
    """

    labels = site_percolation(L, p)[0]

    number_of_clusters = len(np.unique(labels)) - 1
    size = np.array([len(labels[labels == c]) for c in np.unique(labels) if c > 0])

    if len(size) == 0:
        return 0, 0, 0, 0, 0

    max_cluster_size = np.max(size)

    average_cluster_size = np.mean(size)
    sigma_of_size = np.std(size)
    fluctuation_of_size = np.sum(size**2) - max_cluster_size**2
    number_of_clusters = len(np.unique(labels))

    return max_cluster_size, average_cluster_size, sigma_of_size, fluctuation_of_size, number_of_clusters
    

p_range = np.linspace(0, 1, 100)

list_L = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]


list_L = np.arange(600, 1000, 10)

p_range = np.linspace(0.45, 0.65, 100)



def measure_system_2D_parallel(L, p_range, num_processes=30):
    with Pool(num_processes) as p:
        results = p.starmap(measure_system_2D, [(L,p) for p in p_range])
    max_cluster_size, average_cluster_size, sigma_of_size, fluctuation_of_size, number_of_clusters = zip(*results)

    return np.array(max_cluster_size), np.array(fluctuation_of_size)


for L in list_L:
    t1 = time()
    output_file = f'close_crit_percolation_3d_{L}.txt'

    max_cluster_size, fluctuation_of_size = measure_system_2D_parallel(L, p_range)

    np.savetxt(output_file, (max_cluster_size, fluctuation_of_size))
    print(f'Finished L = {L} in {time() - t1} seconds')

