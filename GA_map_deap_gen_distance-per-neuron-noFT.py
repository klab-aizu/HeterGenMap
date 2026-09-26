from HeterGenMap.TaskGraph import TaskGraph
from HeterGenMap.Task import Task
from HeterGenMap.Utils import Coordinate
from HeterGenMap.SystemX import SystemX
import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

testcase = 1
NoC_dim = 3
pop_size = 1
n_gen = 1
n_toursel_mem = 1
n_init_mem = 0 
extra_AER_index = 5


enable_plot = False

if testcase == 0:
    #  MLP - N-MNIST
    TG = TaskGraph("MLP", 15010, [10000,  5000,   10], 2312) # 2312,10000,5000,10
    pop_size = 100
    n_gen = 40
    n_toursel_mem = 5
    extra_AER_index = 5
    NoC_dim = 2

    if NoC_dim == 3:
        s = SystemX(Coordinate(4, 4, 4), 256) # 
    elif NoC_dim == 2:
        s = SystemX(Coordinate(1, 8, 8), 256) # 
    else:
        print ("unsupported testcase!")
        exit()


elif testcase == 1:
    #  MLP - MNIST F_MNIST
    TG = TaskGraph("MLP", 15010, [10000,  5000,   10], 784) #784,10000,5000,10
    pop_size = 100
    n_gen = 40
    n_toursel_mem = 5
    extra_AER_index = 5
    NoC_dim = 2

    if NoC_dim == 3:
        s = SystemX(Coordinate(4, 4, 4), 256) # 
    elif NoC_dim == 2:
        s = SystemX(Coordinate(1, 8, 8), 256) # 
    else:
        print ("unsupported testcase!")
        exit()
else:
    print ("unsupported testcase!")
    exit()

print(">> Perform without FT")

s.gen_routing_cost()

s.naive_assigment(TG.n_tasks)

finalSol1, finalFitness1, log1 =  TG.assign_address("GA_DEAP", s,  i_algo_type="eaMuPlusLambda", i_pop_size=pop_size, i_n_generations=n_gen, i_n_members_in_toursel=n_toursel_mem, n_init_mem=n_init_mem)

def noc_distance(a, b):
    return (
        abs(a["z"] - b["z"]) +
        abs(a["y"] - b["y"]) +
        abs(a["x"] - b["x"])
    )

def calculate_neuron_distances(neurons):
    """
    Calculate NoC traveling distance for each neuron.

    Each source neuron communicates with all neurons
    in the following MLP layer.

    IMPORTANT:
    If multiple destination neurons are located in the
    same NoC PE, they are treated as ONE spike transfer.

    Distance is calculated between unique destination PEs.
    """

    # Group neurons by layer
    layers = {}

    for neuron in neurons:
        layer = neuron["layer"]

        if layer not in layers:
            layers[layer] = []

        layers[layer].append(neuron)

    results = {}

    for layer in sorted(layers.keys()):

        # Last layer has no communication
        if layer + 1 not in layers:
            continue

        current_layer = layers[layer]
        next_layer = layers[layer + 1]

        for src in current_layer:

            # --------------------------------------------------
            # Find UNIQUE destination PEs
            # --------------------------------------------------

            destination_pes = set()

            for dst in next_layer:

                dst_pe = (
                    dst["z"],
                    dst["y"],
                    dst["x"]
                )

                destination_pes.add(dst_pe)

            # --------------------------------------------------
            # Calculate distance to each unique PE
            # --------------------------------------------------

            distances = []

            for dst_pe in destination_pes:

                dst_z, dst_y, dst_x = dst_pe

                distance = (
                    abs(src["z"] - dst_z) +
                    abs(src["y"] - dst_y) +
                    abs(src["x"] - dst_x)
                )

                distances.append({
                    "dst_pe": dst_pe,
                    "distance": distance
                })

            # --------------------------------------------------
            # Total traveling distance
            # --------------------------------------------------

            total_distance = sum(
                d["distance"]
                for d in distances
            )

            average_distance = (
                total_distance / len(distances)
                if len(distances) > 0
                else 0
            )

            results[src["id"]] = {
                "src_id": src["id"],
                "layer": layer,
                "address": (
                    src["z"],
                    src["y"],
                    src["x"]
                ),
                "next_layer": layer + 1,

                # Number of unique destination PEs
                "num_destinations": len(destination_pes),

                "total_distance": total_distance,

                "average_distance": average_distance,

                # Individual PE distances
                "destinations": distances
            }

    return results

def extract_neuron_mapping(A):
    """
    A shape:
        (1, Z, Y, X, L)

    A[0,z,y,x,l] = number of neurons from MLP layer l
                      mapped to NoC node (z,y,x)

    Returns:
        neurons: list of dictionaries containing the address of
                 every neuron.
    """

    # Ignore first dimension
    A = A[0]

    Z, Y, X, L = A.shape

    neurons = []
    neuron_id = 0

    for l in range(L):
        for z in range(Z):
            for y in range(Y):
                for x in range(X):

                    n_neurons = int(A[z, y, x, l])

                    for local_id in range(n_neurons):

                        neurons.append({
                            "id": neuron_id,
                            "layer": l,
                            "z": z,
                            "y": y,
                            "x": x,
                            "local_id": local_id
                        })

                        neuron_id += 1

    return neurons

neurons1 = extract_neuron_mapping(finalSol1)

for n in neurons1[200:270]:
    print(n)

results = calculate_neuron_distances(neurons1)

r = results[10]

print("Neuron:", r["src_id"])
print("Layer:", r["layer"])
print("Address:", r["address"])
print("Next layer:", r["next_layer"])
print("Number of destinations:", r["num_destinations"])
print("Total distance:", r["total_distance"])
print("Average distance:", r["average_distance"])


# get extra AER compression route

import math


def manhattan_distance(a, b):
    """
    Manhattan distance between two 3D NoC coordinates.
    """

    return (
        abs(a[0] - b[0]) +
        abs(a[1] - b[1]) +
        abs(a[2] - b[2])
    )


def get_layer_pes(neurons, layer):
    """
    Return unique NoC PEs containing neurons
    from the specified layer.
    """

    pes = set()

    for neuron in neurons:

        if neuron["layer"] == layer:

            pe = (
                neuron["z"],
                neuron["y"],
                neuron["x"]
            )

            pes.add(pe)

    return list(pes)
def calculate_route(start_pe, pes):
    """
    Calculate a route visiting all PEs.

    Uses a nearest-neighbor heuristic.
    """

    if not pes:
        return [], 0

    unvisited = set(pes)

    # Do not visit the starting PE twice
    unvisited.discard(start_pe)

    current = start_pe

    route = [current]

    total_distance = 0

    while unvisited:

        next_pe = min(
            unvisited,
            key=lambda pe: manhattan_distance(current, pe)
        )

        distance = manhattan_distance(
            current,
            next_pe
        )

        total_distance += distance

        route.append(next_pe)

        current = next_pe

        unvisited.remove(next_pe)

    return route, total_distance

def calculate_neuron_distances_dst_comp(neurons):
    """
    For each neuron:

    1. Start at the neuron's PE.
    2. Visit all unique PEs in the current layer.
    3. Visit all unique PEs in the next layer.
    4. Calculate total Manhattan traveling distance.

    Returns results indexed by Neuron_ID.
    """

    # --------------------------------------------------------
    # Group neurons by layer
    # --------------------------------------------------------

    layers = {}

    for neuron in neurons:

        layer = neuron["layer"]

        if layer not in layers:
            layers[layer] = []

        layers[layer].append(neuron)

    # --------------------------------------------------------
    # Get unique PEs for each layer
    # --------------------------------------------------------

    layer_pes = {}

    for layer in layers:

        layer_pes[layer] = get_layer_pes(
            neurons,
            layer
        )

    results = {}

    # --------------------------------------------------------
    # Process each neuron
    # --------------------------------------------------------

    for layer in sorted(layers.keys()):

        # Last layer has no next layer
        if layer + 1 not in layers:
            continue

        current_layer_pes = layer_pes[layer]

        next_layer_pes = layer_pes[layer + 1]

        for src in layers[layer]:

            source_pe = (
                src["z"],
                src["y"],
                src["x"]
            )

            # ------------------------------------------------
            # Visit current-layer PEs
            # ------------------------------------------------

            route_current, distance_current = calculate_route(
                source_pe,
                current_layer_pes
            )

            # ------------------------------------------------
            # Visit next-layer PEs
            # ------------------------------------------------

            # Continue from the last PE of current-layer route
            current_position = route_current[-1]

            route_next, distance_next = calculate_route(
                current_position,
                next_layer_pes
            )

            # ------------------------------------------------
            # Combine routes
            # ------------------------------------------------

            full_route = (
                route_current +
                route_next[1:]
            )

            total_distance = (
                distance_current +
                distance_next
            )

            results[src["id"]] = {

                "src_id": src["id"],

                "layer": layer,

                "address": source_pe,

                "next_layer": layer + 1,

                "num_current_pes": len(
                    current_layer_pes
                ),

                "num_next_pes": len(
                    next_layer_pes
                ),

                "distance_current_layer":
                    distance_current,

                "distance_next_layer":
                    distance_next,

                "total_distance":
                    total_distance,

                "route":
                    full_route
            }

    return results

result_comp = calculate_neuron_distances_dst_comp(neurons1)

# Find maximum traveling distance for each layer
max_distance_by_layer = {}

for neuron_id, result in result_comp.items():

    layer = result["layer"]
    distance = result["total_distance"]

    if layer not in max_distance_by_layer:
        max_distance_by_layer[layer] = distance
    else:
        max_distance_by_layer[layer] = max(
            max_distance_by_layer[layer],
            distance
        )


# Print results
print("==========================================")
print("Maximum Traveling Distance by MLP Layer")
print("==========================================")

for layer in sorted(max_distance_by_layer):

    print(
        f"Layer {layer} -> Layer {layer + 1}: "
        f"{max_distance_by_layer[layer]} hops"
    )

print("==========================================")


import csv

# Group results by layer
layers = {}

for neuron_index, result in results.items():

    layer = result["layer"]

    if layer not in layers:
        layers[layer] = []

    layers[layer].append(
        (neuron_index, result["total_distance"])
    )




# Write one CSV file for each layer
neuron_id_offset = 0 # set the starting index

for l, dat in layers.items():
        
    filename = f"layer_{l}_distances.csv"

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Neuron_ID",
            "total_traveling_distance"
        ])

        # write the distance
        for neuron_id, distance in dat:
            writer.writerow([neuron_id-neuron_id_offset, distance])
        # write the extra AER
        for i in range(extra_AER_index):
            writer.writerow([neuron_id-neuron_id_offset+i+1, max_distance_by_layer[l]])
        # reset the neuron_id to zero to fit the index from software
        neuron_id_offset = neuron_id
        
