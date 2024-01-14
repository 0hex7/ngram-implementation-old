import numpy as np
from datetime import timedelta
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary
import random

class TrajectoryPerturbation:
    def __init__(self, trajectory, W_set, reachability_function):
        self.trajectory = trajectory
        self.W_set = W_set
        self.reachability_function = reachability_function

    def main_perturbation(self, n, overlap):
        perturbed_trajectory = []

        for i in range(len(self.trajectory)):
            for j in range(i + 1, min(i + overlap + 1, len(self.trajectory) + 1)):
                perturbation_probability = self.calculate_perturbation_probability(i, j, n)
                perturbed_region = self.sample_perturbed_region(perturbation_probability)
                perturbed_trajectory.append(perturbed_region)

        return perturbed_trajectory

    def supplementary_perturbation(self, n, overlap):
        supplementary_perturbed_trajectory = []

        for i in range(1, len(self.trajectory) - 1):
            for j in range(i + 1, min(i + overlap + 1, len(self.trajectory))):
                perturbation_probability = self.calculate_perturbation_probability(i, j, n)
                perturbed_region = self.sample_perturbed_region(perturbation_probability)
                supplementary_perturbed_trajectory.append(perturbed_region)

        return supplementary_perturbed_trajectory

    def calculate_perturbation_probability(self, start_index, end_index, n):
        total_probability = 1.0

        for k in range(start_index, end_index):
            perturbed_set_size = len(self.W_set)
            perturbation_distance = self.calculate_perturbation_distance(start_index, end_index, self.W_set[k])
            perturbation_probability = np.exp(-perturbation_distance / (2 * n**2))
            total_probability *= perturbation_probability / perturbed_set_size

        return total_probability

    def calculate_perturbation_distance(self, start_index, end_index, perturbed_set):
        distance = 0

        for i in range(start_index, end_index):
            original_region = self.trajectory[i]
            perturbed_region = perturbed_set[i - start_index]
            distance += self.reachability_function(original_region, perturbed_region)

        return distance

    def sample_perturbed_region(self, perturbation_probability):
        # Use the EM algorithm to sample from the perturbed set
        sampled_index = np.random.choice(len(self.W_set), p=perturbation_probability)
        return self.W_set[sampled_index]

class POI:
    def __init__(self, location, opening_hours, category, popularity):
        self.location = location
        self.opening_hours = opening_hours
        self.category = category
        self.popularity = popularity

class STCRegion:
    def __init__(self, spatial_region, category, time_interval, pois):
        self.spatial_region = spatial_region
        self.category = category
        self.time_interval = time_interval
        self.pois = pois

def divide_spatial_space(B):
    """
    Divides the physical space into 'B' spatial regions using a uniform grid.

    Parameters:
    - B: Number of spatial regions.

    Returns:
    - List of spatial regions.
    """
    spatial_regions = []
    grid_size = int(B ** 0.5)  # Assuming a square grid for simplicity

    for i in range(grid_size):
        for j in range(grid_size):
            region = {
                'x_min': i * (100 / grid_size),  # Adjust as needed for your coordinate system
                'x_max': (i + 1) * (100 / grid_size),
                'y_min': j * (100 / grid_size),
                'y_max': (j + 1) * (100 / grid_size),
            }
            spatial_regions.append(region)

    return spatial_regions

def create_STC_regions(spatial_regions, POIs, C):
    STC_regions = []

    for spatial_region in spatial_regions:
        for poi_category in set(poi.category for poi in POIs):
            for time_interval in create_coarse_time_intervals_for_category(C):
                pois_in_region = [poi for poi in POIs if poi.location in spatial_region and poi.category == poi_category and poi.opening_hours_overlap(time_interval)]

                if pois_in_region:
                    stc_region = STCRegion(spatial_region, poi_category, time_interval, pois_in_region)
                    STC_regions.append(stc_region)

    return STC_regions

def create_coarse_time_intervals(STC_regions, C):
    for stc_region in STC_regions:
        time_intervals = create_coarse_time_intervals_for_category(C)
        stc_region.time_intervals = time_intervals

    return STC_regions

def create_coarse_time_intervals_for_category(C):
    time_intervals = []
    interval_length = 24 // C

    for i in range(C):
        start_time = i * interval_length
        end_time = start_time + interval_length
        time_intervals.append((start_time, end_time))

    return time_intervals

def remove_empty_regions(STC_regions):
    return [stc_region for stc_region in STC_regions if stc_region.pois]
def region_error_distance(region1, region2, trajectory_indices):
    """
    Calculate the region error distance between two regions.

    Parameters:
    - region1: Coordinates or features of the first region.
    - region2: Coordinates or features of the second region.
    - trajectory_indices: Indices of the trajectory points being compared.

    Returns:
    - Region error distance.
    """
    # Assuming regions are represented as tuples of coordinates, modify this based on your actual data structure
    region1_coords = region1
    region2_coords = region2

    # Extract the trajectory points corresponding to the given indices
    trajectory_points1 = [region1_coords[index] for index in trajectory_indices]
    trajectory_points2 = [region2_coords[index] for index in trajectory_indices]

    # Implement your region error distance function based on the given formula
    # This is a placeholder, replace it with your actual calculation
    region_error_distance = 0.0  # Replace this with your calculation

    return region_error_distance


def bigram_error_distance(bigram, trajectory_indices):
    """
    Calculate the bigram error distance for a given bigram.

    Parameters:
    - bigram: A region-level bigram (F) from W2.
    - trajectory_indices: Indices of the trajectory points being compared.

    Returns:
    - Bigram error distance.
    """
    # Assuming bigrams are represented as tuples of regions, modify this based on your actual data structure
    region1, region2 = bigram

    # Calculate region error distances for the two regions in the bigram
    region_error1 = region_error_distance(region1, region2, trajectory_indices)
    region_error2 = region_error_distance(region2, region1, trajectory_indices)  # Assuming symmetric

    # Implement your bigram error distance function based on the given formula
    # This is a placeholder, replace it with your actual calculation
    bigram_error_distance = region_error1 + region_error2  # Replace this with your calculation

    return bigram_error_distance


def merge_regions(STC_regions):
    merged_regions = {}

    for stc_region in STC_regions:
        key = (stc_region.spatial_region, stc_region.category)
        if key not in merged_regions:
            merged_regions[key] = stc_region
        else:
            merged_regions[key].pois.extend(stc_region.pois)

    return list(merged_regions.values())

def instantiate_n_grams(STC_regions):
    n_gram_combinations = []

    for i in range(len(STC_regions)):
        for j in range(i + 1, len(STC_regions) + 1):
            n_gram_combinations.append(STC_regions[i:j])

    return n_gram_combinations

def check_reachability(n_gram, reachability_function):
    """
    Check if the given n-gram satisfies the reachability constraint.

    Parameters:
    - n_gram: A specific n-gram combination.
    - reachability_function: The reachability function to apply.

    Returns:
    - True if the n-gram satisfies the reachability constraint, False otherwise.
    """
    # Example: Check if the sum of popularity in the n-gram exceeds a certain threshold
    threshold = 15  # Adjust this threshold as needed
    total_popularity = sum(poi.popularity for stc_region in n_gram for poi in stc_region.pois)
    return total_popularity >= threshold

def filter_valid_n_grams(n_gram_combinations, reachability_function):
    """
    Remove n-gram combinations that do not satisfy reachability constraint.

    Parameters:
    - n_gram_combinations: List of n-gram combinations.
    - reachability_function: Reachability function to apply.

    Returns:
    - List of valid n-gram combinations.
    """
    valid_n_grams = []

    for n_gram in n_gram_combinations:
        if check_reachability(n_gram, reachability_function):
            valid_n_grams.append(n_gram)

    return valid_n_grams

def reachability_function(n_gram_set):
    """
    Reachability function for n-gram sets.

    Parameters:
    - n_gram_set: A specific n-gram set.

    Returns:
    - True if the n-gram set satisfies the reachability constraint, False otherwise.
    """
    # Implement your reachability logic based on the description

    for i in range(len(n_gram_set)):
        for j in range(len(n_gram_set)):
            if i != j:
                # Check if there is at least one region in n_gram_set[i] and one region in n_gram_set[j]
                if any(region_in_i for region_in_i in n_gram_set[i]) and any(region_in_j for region_in_j in n_gram_set[j]):
                    return True

    # If no satisfying pairs are found, return False
    return False

def hierarchical_decomposition(POIs, B, C, reachability_function):
    spatial_regions = divide_spatial_space(B)
    STC_regions = create_STC_regions(spatial_regions, POIs, C)
    STC_regions = create_coarse_time_intervals(STC_regions, C)
    STC_regions = remove_empty_regions(STC_regions)
    STC_regions = merge_regions(STC_regions)
    n_gram_combinations = instantiate_n_grams(STC_regions)
    valid_n_grams = filter_valid_n_grams(n_gram_combinations, reachability_function)
    return valid_n_grams

def trajectory_reconstruction(perturbed_trajectory, perturbed_set):
    # Create a linear programming problem
    reconstruction_problem = LpProblem("Trajectory_Reconstruction", LpMinimize)

    # Define binary variables for each bigram in the perturbed set
    bigram_variables = LpVariable.dicts("Bigram", [(i, j) for i in range(len(perturbed_trajectory)) for j in range(len(perturbed_set))], 0, 1, LpBinary)

    # Define the region error term
    region_error = lpSum([bigram_variables[(i, j)] * region_error_distance(perturbed_set[j][i], perturbed_trajectory[i]) for i in range(len(perturbed_trajectory)) for j in range(len(perturbed_set))])

    # Define the bigram error term
    bigram_error = lpSum([bigram_variables[(i, j)] * bigram_error_distance(perturbed_set[j], perturbed_set[j + 1]) for i in range(len(perturbed_trajectory) - 1) for j in range(len(perturbed_set) - 1)])

    # Objective function to minimize the total error
    reconstruction_problem += region_error + bigram_error

    # Constraints
    for i in range(len(perturbed_trajectory) - 1):
        reconstruction_problem += lpSum([bigram_variables[(i, j)] for j in range(len(perturbed_set))]) == 1  # Each point in the trajectory is associated with exactly one bigram

    for j in range(len(perturbed_set) - 1):
        reconstruction_problem += lpSum([bigram_variables[(i, j)] for i in range(len(perturbed_trajectory))]) == lpSum([bigram_variables[(i, j + 1)] for i in range(len(perturbed_trajectory))])  # Continuity constraint

    # Solve the linear programming problem
    reconstruction_problem.solve()

    # Extract the reconstructed trajectory
    reconstructed_trajectory = [perturbed_set[j][i] for i in range(len(perturbed_trajectory)) for j in range(len(perturbed_set)) if bigram_variables[(i, j)].value() == 1]

    return reconstructed_trajectory

def main():
    # Define your POIs
    poi1 = POI(location=(0, 0), opening_hours=(8, 18), category="Park", popularity=5)
    poi2 = POI(location=(1, 1), opening_hours=(10, 22), category="Restaurant", popularity=8)
    # ... add more POIs ...

    # List of POIs
    POIs = [poi1, poi2]

    # Call the hierarchical decomposition function
    result = hierarchical_decomposition(POIs, B=5, C=4, reachability_function=reachability_function)

    # Print or use the result as needed
    print(result)

    # Assume you have a perturbed_trajectory and a perturbed_set
    perturbed_trajectory = [(1, 2), (2, 3), (3, 4), (4, 5)]
    perturbed_set = [perturbed_trajectory]  # Replace with actual perturbed set

    # Perform trajectory reconstruction
    reconstructed_trajectory = trajectory_reconstruction(perturbed_trajectory, perturbed_set)

    # Print or use the reconstructed_trajectory as needed
    print(reconstructed_trajectory)

    # Assume you have a perturbed_region_sequence and a reachability_constraint
    perturbed_region_sequence = [
        {"poi": ["Restaurant"], "time": ["9-10pm"], "location": ["Downtown"]},
        {"poi": ["Bar"], "time": ["9-10pm"], "location": ["Downtown"]},
        {"poi": ["Bar"], "time": ["9-10pm"], "location": ["Suburb"]}
    ]

    reachability_constraint = 55  # Replace with your actual reachability constraint

    # Perform POI-level trajectory reconstruction
    reconstructed_poi_trajectory = poi_level_trajectory_reconstruction(perturbed_region_sequence, reachability_constraint)

    # Print or use the reconstructed_poi_trajectory as needed
    print(reconstructed_poi_trajectory)

if __name__ == "__main__":
    main()
