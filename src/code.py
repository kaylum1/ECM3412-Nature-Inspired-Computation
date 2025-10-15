"""All that is required to start the porgram is to press Run"""


""" -----------Import and initialising data ---------------------"""

import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt

# Define general parameters
NUM_ITEMS = 500
EVALUATIONS = 10000  # Set evaluations to 10,000





''' -----------set up functions ----------------------------------'''

# Initialize item weights and bins based on BPP setting
def initialize_bpp_parameters(bpp_weight_set):
    if bpp_weight_set == 1:
        num_bins = 10
        item_weights = np.arange(1, NUM_ITEMS + 1)
    elif bpp_weight_set == 2:
        num_bins = 50
        item_weights = ((np.arange(1, NUM_ITEMS + 1) ** 2) / 2)
    return num_bins, item_weights




#this function is what initialises the pheromone matrix with random number between 1 and 0
def initialise_and_distribute_pheromones_matrix(items, bins):
    return np.random.rand(items, bins)


#this function generate paths for ants based on the biased probabilities influenced by pheromone levels
def Generate_ant_path_probabilities(random_matrix, ants):
    biased_matrix = random_matrix / random_matrix.sum(axis=1, keepdims=True)
    paths = np.array([
        choice(np.arange(biased_matrix.shape[1]), p=biased_matrix[i], size=ants)
        for i in range(biased_matrix.shape[0])
    ]).T
    return paths + 1  # Convert to 1-based indexing


#function that calculates the fitness of the all the ant path 
def CalculateFitness(paths_set, items_set, bins_set, ants_set, item_weights):
    paths_set = paths_set - 1  # Convert to zero-based indexing for array compatibility
    bin_weights = np.zeros((ants_set, bins_set))
    
    # section Vectorized all the bin weights calcualtions 
    for ant in range(ants_set):
        bin_weights[ant] = np.bincount(paths_set[ant], weights=item_weights, minlength=bins_set)
    
    # Fitness is the difference between the max and min bin weights
    diff = np.max(bin_weights, axis=1) - np.min(bin_weights, axis=1)
    return diff



""" ----------------main algoirthm --------------------------------"""


# Define the main ACO function that calls all the other function above in the correct order
def AntColonyOptimization(exp_id, bpp_weight_set):
    # Set parameters based on experiment ID
    params = {
        1: (100, 0.9),
        2: (100, 0.6),
        3: (10, 0.9),
        4: (10, 0.6)
    }
    num_ants, evaporation_rate = params.get(exp_id, (100, 0.9))
    
    # Get BPP-specific parameters
    num_bins, item_weights = initialize_bpp_parameters(bpp_weight_set)
    weight_type = "linear" if bpp_weight_set == 1 else "quadratic"
    

    #this is the printing of what experement and the data for evaporation and ants
    print(f"\nRunning Experiment {exp_id} for BPP{bpp_weight_set} (5 trials):")
    print(f"Parameters: Number of Ants = {num_ants}, Evaporation Rate = {evaporation_rate}, Number of Bins = {num_bins}, Bin Weight Type = {weight_type}\n")

    all_trials_results = []  # Store results of all trials for the final overlapping plot
    final_fitness_values = []  # To store final fitness for each trial


    #this inititates the 5 trials needed for each experment of the BPP cases
    for trial in range(5):
        print(f"\n--- Trial {trial + 1} ---")
        


    # Initialize parameters for the specific BPP instance (1 or 2)
        random_matrix = initialise_and_distribute_pheromones_matrix(NUM_ITEMS, num_bins)
        avg_fitness_over_evaluations = []
        
        for itr in range(EVALUATIONS):
            all_paths = Generate_ant_path_probabilities(random_matrix, num_ants)
            diff = CalculateFitness(all_paths, NUM_ITEMS, num_bins, num_ants, item_weights)
            delta_pheromone = 100 / np.min(diff)
            
            # Batch pheromone update for the best path to help speed up the process and it extremly slow before
            best_path = all_paths[np.argmin(diff)]
            update_indices = (np.arange(NUM_ITEMS), best_path - 1)
            pheromone_update = np.zeros_like(random_matrix)
            pheromone_update[update_indices] += delta_pheromone
            random_matrix = random_matrix * evaporation_rate + pheromone_update

            # Calculate fitness metrics
            avg_fitness = np.mean(diff)
            best_fitness = np.min(diff)
            worst_fitness = np.max(diff)
            avg_fitness_over_evaluations.append(avg_fitness)

            if itr == 0 or itr == EVALUATIONS - 1:
                print(f"Evaluation {itr+1}: Avg Fitness = {avg_fitness:.2f}, Best Fitness = {best_fitness:.2f}, Worst Fitness = {worst_fitness:.2f}")

        # Save the average fitness for this trial for final plot
        all_trials_results.append(avg_fitness_over_evaluations)
        final_fitness_values.append(avg_fitness_over_evaluations[-1])  # Store final fitness

    # Final overlapping plot for all trials
    plt.figure()
    for trial_num, trial_results in enumerate(all_trials_results, 1):
        plt.plot(trial_results, label=f'Trial {trial_num}')
    plt.xlabel('Evaluations')
    plt.ylabel('Average Fitness')
    plt.title(f'Overlapping Average Fitness over Evaluations - Experiment {exp_id}, BPP{bpp_weight_set}')
    plt.legend()
    plt.show()

    # Overlapping plot showing only the first 400 evaluations for all trials
    plt.figure()
    for trial_num, trial_results in enumerate(all_trials_results, 1):
        plt.plot(trial_results[:400], label=f'Trial {trial_num}')
    plt.xlabel('Evaluations (First 400)')
    plt.ylabel('Average Fitness')
    plt.title(f'Overlapping Average Fitness (First 400 Evaluations) - Experiment {exp_id}, BPP{bpp_weight_set}')
    plt.legend()
    plt.show()
    
    # Return the average of all trials' fitness values
    avg_of_all_trials = np.mean(all_trials_results, axis=0)
    return avg_of_all_trials, final_fitness_values

# Run the experiments and store the final fitness values for box plots
box_plot_data = []
experiment_labels = []


"""--------------------------this is the start of experiments for BPP1 -----------------------------------"""

#this will run five trials of the ACO with p = 100 and e = 0.90

avg_results_exp1, final_fitness_exp1 = AntColonyOptimization(1, 1)
box_plot_data.append(final_fitness_exp1)
experiment_labels.append("Experiment 1")


#this will Run five trials of the ACO with p = 100 and e = 0.60


avg_results_exp2, final_fitness_exp2 = AntColonyOptimization(2, 1)
box_plot_data.append(final_fitness_exp2)
experiment_labels.append("Experiment 2")


#this will Run five trials of the ACO with p = 10, and e = 0.90


avg_results_exp3, final_fitness_exp3 = AntColonyOptimization(3, 1)
box_plot_data.append(final_fitness_exp3)
experiment_labels.append("Experiment 3")


# this Run five trials of the ACO with p = 10, and e = 0.60


avg_results_exp4, final_fitness_exp4 = AntColonyOptimization(4, 1)
box_plot_data.append(final_fitness_exp4)
experiment_labels.append("Experiment 4")


# Final plot of averages from each experiment which will overlay all the BPP1 scores on a graph
plt.figure()
plt.plot(avg_results_exp1, label="ACO with p=100, e=0.90")
plt.plot(avg_results_exp2, label="ACO with p=100, e=0.60")
plt.plot(avg_results_exp3, label="ACO with p=10, e=0.90")
plt.plot(avg_results_exp4, label="ACO with p=10, e=0.60")
plt.xlabel('Evaluations')
plt.ylabel('Average Final Fitness')
plt.title('Average Final Fitness Over Evaluations for BPP1')
plt.legend()
plt.show()




# Combined box plot for final fitness values across experiments
plt.figure(figsize=(10, 6))
plt.boxplot(box_plot_data, labels=experiment_labels)
plt.xlabel('Experiment')
plt.ylabel('Final Fitness')
plt.title('Box Plot of Final Fitness Across Trials for Each Experiment (BPP1)')
plt.show()




"""--------------------------this is the start of experiments for BPP2 -----------------------------------"""


#this will run five trials of the ACO with p = 100 and e = 0.90

avg_results_exp5, final_fitness_exp5 = AntColonyOptimization(1, 2)
box_plot_data.append(final_fitness_exp5)
experiment_labels.append("Experiment 5")


#this will Run five trials of the ACO with p = 100 and e = 0.60

avg_results_exp6, final_fitness_exp6 = AntColonyOptimization(2, 2)
box_plot_data.append(final_fitness_exp6)
experiment_labels.append("Experiment 6")


#this will Run five trials of the ACO with p = 10, and e = 0.90

avg_results_exp7, final_fitness_exp7 = AntColonyOptimization(3, 2)
box_plot_data.append(final_fitness_exp7)
experiment_labels.append("Experiment 7")



#this will Run five trials of the ACO with p = 10, and e = 0.60

avg_results_exp8, final_fitness_exp8 = AntColonyOptimization(4, 2)
box_plot_data.append(final_fitness_exp8)
experiment_labels.append("Experiment 8")



# Final plot of averages from each experiment which will overlay all the BPP2 scores on a graph
plt.figure()
plt.plot(avg_results_exp5, label="ACO with p=100, e=0.90")
plt.plot(avg_results_exp6, label="ACO with p=100, e=0.60")
plt.plot(avg_results_exp7, label="ACO with p=10, e=0.90")
plt.plot(avg_results_exp8, label="ACO with p=10, e=0.60")
plt.xlabel('Evaluations')
plt.ylabel('Average Final Fitness')
plt.title('Average Final Fitness Over Evaluations for BPP2')
plt.legend()
plt.show()



# this is a combined box plot for final fitness values across experiments which will have BPP1 and BPP@
plt.figure(figsize=(10, 6))
plt.boxplot(box_plot_data, labels=experiment_labels)
plt.xlabel('Experiment')
plt.ylabel('Final Fitness')
plt.title('Box Plot of Final Fitness Across Trials for Each Experiment (BPP2)')
plt.show()