import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from multiprocessing import Pool

def site_percolation(L: int, p: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate site percolation on a 3D lattice of size L x L x L with site
    occupation probability p using the Hoshen-Kopelman algorithm.

    Parameters:
        L (int): The size of the lattice.
        p (float): The site occupation probability.

    Returns:
        labels (np.ndarray): A 3D array containing the labels of each cluster.
        lattice (np.ndarray): A 3D array containing the occupation state of each site.
    """

    # Generate a random L x L x L lattice and set sites as occupied or unoccupied based on p
    rand_lattice = np.random.rand(L, L, L)
    lattice = rand_lattice < p
    
    # Initialize the labels array to zeros
    labels = np.zeros((L, L, L), dtype=int)

    # Initialize the label counter to 1
    label_counter = 1

    # Iterate over all sites in the lattice
    for i in range(L):
        for j in range(L):
            for k in range(L):

                # If the site is occupied, assign it a label and update the label counter
                if lattice[i, j, k]:
                    neighbors = []

                    # Check if the neighboring sites are occupied and have a label
                    if i > 0 and lattice[i-1, j, k]:
                        neighbors.append(labels[i-1, j, k])
                    if j > 0 and lattice[i, j-1, k]:
                        neighbors.append(labels[i, j-1, k])
                    if k > 0 and lattice[i, j, k-1]:
                        neighbors.append(labels[i, j, k-1])

                    # If the site has no neighbors, assign it a new label
                    if not neighbors:
                        labels[i, j, k] = label_counter
                        label_counter += 1

                    # If the site has neighbors, assign it the label of its lowest numbered neighbor
                    else:
                        neighbors = np.unique(neighbors)
                        labels[i, j, k] = neighbors[0]

                        # Merge clusters with the same label
                        for neighbor in neighbors[1:]:
                            labels[labels == neighbor] = labels[i, j, k]

    return labels, np.array(lattice, dtype=int)


# function to measure the percolation properties of a 2D system
def measure_system_2D(L, p):

    # obtain the labels from site percolation
    labels = site_percolation(L, p)[0]

    # calculate the number of clusters and their size distribution
    number_of_clusters = len(np.unique(labels)) - 1
    size = np.array([len(labels[labels == c]) for c in np.unique(labels) if c > 0])

    # return 0 for both properties if no clusters are present
    if len(size) == 0:
        return 0, 0

    # calculate the maximum cluster size and the fluctuation of cluster size distribution
    max_cluster_size = np.max(size)
    fluctuation_of_size = np.sum(size**2) - max_cluster_size**2

    # return the maximum cluster size and fluctuation of cluster size distribution
    return max_cluster_size, fluctuation_of_size

# generate a range of probabilities to simulate percolation
p_range = np.linspace(0.25, 0.35, 100)

# generate a range of system sizes to measure percolation properties
list_L = np.arange(10, 100, 10)

# function to measure the percolation properties of a 2D system in parallel
def measure_system_2D_parallel(L, p_range, num_processes=30):

    # perform parallel processing using the measure_system_2D function
    with Pool(num_processes) as p:
        results = p.starmap(measure_system_2D, [(L,p) for p in p_range])

    # separate the maximum cluster size and the fluctuation of cluster size distribution
    max_cluster_size, fluctuation_of_size, = zip(*results)

    # return the maximum cluster size and fluctuation of cluster size distribution as numpy arrays
    return np.array(max_cluster_size), np.array(fluctuation_of_size)


# Define the range of lattice sizes
list_of_L = np.arange(10, 100, 10)

# Define the range of probabilities
p_range = np.linspace(0.25, 0.35, 100)


def save_data(L, p_range):
    """
    Measure system properties for a given lattice size and range of probabilities,
    and save the data to a file.
    """
    t1 = time()
    output_file = f'3d_perc2/close_crit_percolation_3d_{L}.txt'

    # Measure the system properties in parallel
    max_cluster_size, fluctuation_of_size = measure_system_2D_parallel(L, p_range)

    # Save the data to a file
    np.savetxt(output_file, (max_cluster_size, fluctuation_of_size))

    print(f'Finished L = {L} in {time() - t1} seconds')

# Loop over the range of lattice sizes and save the data for each
for L in list_of_L:
    save_data(L, p_range)
