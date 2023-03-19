using Distributed

@everywhere using Statistics, Random, DelimitedFiles


function site_percolation(L, p)
    """
    Site percolation in a square lattice of size L x L with probability p.
    Returns the percolation map and the number of clusters.
    """
    # Generate a lattice with random site occupation probabilities
    lattice = rand(Float64, L, L, L) .< p
    
    # Initialize the label array
    labels = zeros(Int64, L, L, L)

    # Assign labels to the clusters
    label = 1
    for i in 1:L
        for j in 1:L
            for k in 1:L
                if lattice[i, j, k]
                    neighbors = []
                    if i > 1 && lattice[i-1, j, k]
                        push!(neighbors, labels[i-1, j, k])
                    end
                    if j > 1 && lattice[i, j-1, k]
                        push!(neighbors, labels[i, j-1, k])
                    end
                    if k > 1 && lattice[i, j, k-1]
                        push!(neighbors, labels[i, j, k-1])
                    end
                    if isempty(neighbors)
                        labels[i, j, k] = label
                        label += 1
                    else
                        neighbors = unique(neighbors)
                        labels[i, j, k] = neighbors[1]
                        for neighbor in neighbors[2:end]
                            labels[labels .== neighbor] .= labels[i, j, k]
                        end 
                    end
                end
            end
        end
    end

    return labels
end




function measure_system_3D(L::Int, p_range::AbstractVector{Float64})
    """
    Measures the basic quantities of the 2D site percolation system with lattice size L
    for site occupation probabilities in the range p_range.
    Returns the number of clusters, the largest cluster size, and the percolation probability
    for each probability in p_range.
    """

    results = [(site_percolation(L, p),) for p in p_range]
    max_cluster_size = []
    fluctuation_of_size = []


    for res in results
        labels = res[1]

        number_of_clusters = length(unique(labels))
        cluster_sizes = [length(labels[labels .== c]) for c in unique(labels) if c > 0]
    
        if length(cluster_sizes) == 0
            push!(fluctuation_of_size, 0)
            push!(max_cluster_size, 0)
        else
            push!(fluctuation_of_size, sum(cluster_sizes.^2) - maximum(cluster_sizes)^2)
            push!(max_cluster_size, maximum(cluster_sizes))
        end 
    end

    return max_cluster_size, fluctuation_of_size
end


list_L = range(10, stop=1000, step=10)

p_range = range(0.25, 0.35, length=100)

# Run the simulation for each L in list_L and save the results to file
for L in list_L
    @time begin
        output_file = "../PhaseTransitions/julia_txt/percolation_3d_$(L).txt"
        res = measure_system_3D(L, p_range)
        writedlm(output_file, res, ',')
        println("Finished L = $L")
    end 
end
