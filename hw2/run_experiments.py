# Script to run experiments
# https://github.com/ezerilli/Machine_Learning/tree/master/Randomized_Optimization
import neural_networks as nn
import numpy as np
import matplotlib.pyplot as plt
import randomized_optimization as ro
import sys
# sys.path.inster(mlrose)

import pandas as pd
from mlrose.opt_probs import TSPOpt, DiscreteOpt
from mlrose.fitness import TravellingSales, FlipFlop, FourPeaks, Queens, MaxKColor, Knapsack, SixPeaks

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def pre_process1(csv_file, noise):
    # https://stackoverflow.c"'
    # x, y, labels, features = data.data, data.target, data.target_names, data.feature_names
    data = pd.read_csv(csv_file)
    headers = list(data.columns)
    if not not noise:
        # is not empty.
        noisy_data = data[headers[1:(len(headers) - 1)]]
        noisy_header = headers[1:(len(headers) - 1)]
        noisy_shape = np.shape(noisy_data)
        noisy_variables = int(noisy_shape[0] * noisy_shape[1] * noise)
        noisyx = np.random.randint(0, noisy_shape[0], noisy_variables)
        noisyy = np.random.randint(0, noisy_shape[1], noisy_variables)
        for y in noisyy:
            data.loc[noisyx[noisyy == y], noisy_header[y]] = 'NA'

    vals = data[headers[:-1]].stack().drop_duplicates().values
    b = [x for x in data[headers[:-1]].stack().drop_duplicates().rank(method='dense')]
    dictionary = dict(zip(b, vals))  # dictionary for digitization.

    stacked = data[headers[:-1]].stack()
    data[headers[:-1]] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()

    # class_vals = data[headers[-1]].drop_duplicates().values
    # b = data[headers[-1]].drop_duplicates().rank(method='dense')
    data[headers[-1]] = pd.Series(data[headers[-1]].factorize()[0])

    return data, dictionary

def load_dataset(split_percentage=0.2, dataset1=[], data=load_wine()):
    """Load WDBC dataset.

        Args:
           split_percentage (float): validation split.

        Returns:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           y_test (ndarray): test labels.
        """

    # Loadt dataset and split in training and validation sets, preserving classes representation

    if not dataset1:
        data = load_breast_cancer()
        x, y, labels, features = data.data, data.target, data.target_names, data.feature_names
    else:
        # dataset1 = "diabetes_data_upload.csv"
        data, dictionary = pre_process1(dataset1, noise=False)
        x = data.iloc[:, 0:-1]
        y = np.asarray(data.iloc[:, -1])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_percentage, shuffle=True, random_state=42, stratify=y)

    # Normalize feature data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print('\nTotal dataset size:')
    print('Number of instances: {}'.format(x.shape[0]))
    print('Number of features: {}'.format(x.shape[1]))
    # print('Number of classes: {}'.format(len(labels)))
    print('Training Set : {}'.format(x_train.shape))
    print('Testing Set : {}'.format(x_test.shape))

    return x_train, x_test, y_train, y_test


def max_color(length, edges, random_seeds):
    """Define and experiment Max Color optimization problem.
        Fitness function for Max-k color optimization problem. Evaluates the
        fitness of an n-dimensional state vector
        :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]`, where :math:`x_{i}`
        represents the color of node i, as the number of pairs of adjacent nodes
        of the same color.
        Args:
           length (int): problem length.
           random_seeds (list or ndarray): random seeds for get performances over multiple random runs.

        Returns:
           None.
        """

    # Define k max color objective function and problem
    maxkcolor_prob = MaxKColor(edges=edges)
    problem = DiscreteOpt(length=length, fitness_fn=maxkcolor_prob, maximize=True, max_val=2)
    problem.mimic_speed = True

    # Plot optimizations for RHC, SA, GA and MIMIC
    print('\nPlot Optimizations for SA, GA and MIMIC')
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=700, sa_max_iters=700, ga_max_iters=250, mimic_max_iters=50,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.02, 0.1, 0.02), sa_min_temp=0.001,
                          ga_pop_size=300, mimic_pop_size=1000, ga_keep_pct=0.2, mimic_keep_pct=0.4,
                          pop_sizes=np.arange(200, 1001, 200), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='Max Color', plot_ylabel='Fitness')

    # Plot performances for RHC, SA, GA and MIMIC
    print('\nPlot Optimizations for RHC, SA, GA and MIMIC')
    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=700, sa_max_iters=700, ga_max_iters=500, mimic_max_iters=100,
                         sa_init_temp=100, sa_exp_decay_rate=0.04, sa_min_temp=0.001,
                         ga_pop_size=300, ga_keep_pct=0.2,
                         mimic_pop_size=1500, mimic_keep_pct=0.4,
                         plot_name='Max Color', plot_ylabel='Fitness')


def queens(length, random_seeds):
    """Define and experiment Queens optimization problem.

        Fitness function for N-Queens optimization problem. Evaluates the
        fitness of an n-dimensional state vector
        :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]`, where :math:`x_{i}`
        represents the row position (between 0 and n-1, inclusive) of the 'queen'
        in column i, as the number of pairs of attacking queens.

        Args:
           length (int): problem length.
           random_seeds (list or ndarray): random seeds for get performances over multiple random runs.

        Returns:
           None.
        """

    # Define Queens objective function and problem
    queen = Queens()
    problem = DiscreteOpt(length=length, fitness_fn=queen, maximize=True, max_val=2)
    problem.mimic_speed = True  # set fast MIMIC

    # Plot optimizations for RHC, SA, GA and MIMIC
    print('\nPlot Optimizations for SA, GA and MIMIC')
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=7000, sa_max_iters=7000, ga_max_iters=250, mimic_max_iters=250,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.02, 0.1, 0.02), sa_min_temp=0.001,
                          ga_pop_size=1000, mimic_pop_size=1000, ga_keep_pct=0.1, mimic_keep_pct=0.2,
                          pop_sizes=np.arange(200, 1001, 200), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='Queen', plot_ylabel='Fitness')

    # Plot performances for RHC, SA, GA and MIMIC
    print('\nPlot Optimizations for RHC, SA, GA and MIMIC')
    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=7000, sa_max_iters=7000, ga_max_iters=250, mimic_max_iters=250,
                         sa_init_temp=100, sa_exp_decay_rate=0.02, sa_min_temp=0.001,
                         ga_pop_size=1000, ga_keep_pct=0.1,
                         mimic_pop_size=1000, mimic_keep_pct=0.2,
                         plot_name='Queen', plot_ylabel='Fitness')

def four_peaks(length, random_seeds):
    """Define and experiment Four Peaks optimization problem.
        Fitness function for Four Peaks optimization problem. Evaluates the
        fitness of an n-dimensional state vector :math:`x`, given parameter T, as:

        .. math::
            Fitness(x, T) = \\max(tail(0, x), head(1, x)) + R(x, T)

        where:

        * :math:`tail(b, x)` is the number of trailing b's in :math:`x`;
        * :math:`head(b, x)` is the number of leading b's in :math:`x`;
        * :math:`R(x, T) = n`, if :math:`tail(0, x) > T` and
          :math:`head(1, x) > T`; and
        * :math:`R(x, T) = 0`, otherwise.
        Args:
           length (int): problem length.
           random_seeds (list or ndarray): random seeds for get performances over multiple random runs.
        Returns:
           None.
        """

    # Define Four Peaks objective function and problem
    four_fitness = FourPeaks(t_pct=0.1)
    problem = DiscreteOpt(length=length, fitness_fn=four_fitness, maximize=True, max_val=2)
    problem.mimic_speed = True  # set fast MIMIC

    # Plot optimizations for RHC, SA, GA and MIMIC
    print('\nPlot Optimizations for SA, GA and MIMIC')
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=7000, sa_max_iters=7000, ga_max_iters=500, mimic_max_iters=250,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.02, 0.1, 0.02), sa_min_temp=0.001,
                          ga_pop_size=1000, mimic_pop_size=1000, ga_keep_pct=0.1, mimic_keep_pct=0.2,
                          pop_sizes=np.arange(200, 1001, 200), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='Four Peaks', plot_ylabel='Fitness')

    # Plot performances for RHC, SA, GA and MIMIC
    print('\nPlot Optimizations for RHC, SA, GA and MIMIC')
    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=7000, sa_max_iters=7000, ga_max_iters=250, mimic_max_iters=250,
                         sa_init_temp=100, sa_exp_decay_rate=0.02, sa_min_temp=0.001,
                         ga_pop_size=1000, ga_keep_pct=0.1,
                         mimic_pop_size=1000, mimic_keep_pct=0.2,
                         plot_name='Four Peaks', plot_ylabel='Fitness')

def travel_salesman(length, coords, random_seeds):
    """Define and experiment the Travel Salesman optimization problem.
        Fitness function for Travelling Salesman optimization problem.
        Evaluates the fitness of a tour of n nodes, represented by state vector
        :math:`x`, giving the order in which the nodes are visited, as the total
        distance travelled on the tour (including the distance travelled between
        the final node in the state vector and the first node in the state vector
        during the return leg of the tour). Each node must be visited exactly
        once for a tour to be considered valid.
        Args:
           length (int): problem length.
           distances(list of tuples): list of inter-distances between each pair of cities.
           random_seeds (list or ndarray): random seeds for get performances over multiple random runs.
        Returns:
           None.
        """

    # Define Travel Salesman objective function and problem
    tsp_objective = TravellingSales(coords = coords)
    problem = TSPOpt(length=length, fitness_fn=tsp_objective, maximize=True)
    problem.mimic_speed = True  # set fast MIMIC

    # Plot optimizations for SA, GA and MIMIC
    print('\nPlot Optimizations for SA, GA and MIMIC')
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=500, sa_max_iters=500, ga_max_iters=50, mimic_max_iters=10,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.005, 0.105, 0.005), sa_min_temp=0.001,
                          ga_pop_size=100, mimic_pop_size=700, ga_keep_pct=0.2, mimic_keep_pct=0.2,
                          pop_sizes=np.arange(400, 1601, 400), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='TSP', plot_ylabel='Fitness')

    # Plot performances for RHC, SA, GA and MIMIC
    print('\nPlot Optimizations for RHC, SA, GA and MIMIC')
    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=500, sa_max_iters=500, ga_max_iters=50, mimic_max_iters=10,
                         sa_init_temp=100, sa_exp_decay_rate=0.03, sa_min_temp=0.001,
                         ga_pop_size=100, ga_keep_pct=0.2,
                         mimic_pop_size=700, mimic_keep_pct=0.2,
                         plot_name='TSP', plot_ylabel='Fitness')

def neural_network(x_train, x_test, y_train, y_test, random_seeds):
    """Define and experiment the Neural Network weights optimization problem.

        Training Neural Networks weights can be done using GD and backpropation, but also another RO
        optimization algorithm, like RHC, SA or GA, can be used.

        Args:
          x_train (ndarray): training data.
          x_test (ndarray): test data.
          y_train (ndarray): training labels.
          y_test (ndarray): test labels.
          random_seeds (list or ndarray): random seeds for get performances over multiple random runs.

        Returns:
          None.
        """
    # Maximum iterations to run the Neural Network for
    iterations = np.array([i for i in range(1, 10)] + [10 * i for i in range(1, 20, 2)])

    # Plot performances for RHC, SA, GA and GD with Neural Networks
    nn.plot_nn_performances(x_train, y_train,
                            random_seeds=random_seeds,
                            rhc_max_iters=iterations, sa_max_iters=iterations,
                            ga_max_iters=iterations, gd_max_iters=iterations, #mimic_max_iters=iterations,
                            init_temp=100, exp_decay_rate=0.1, min_temp=0.001,
                            pop_size=100, mutation_prob=0.2)

    # Test performances for RHC, SA, GA and GD with Neural Networks
    nn.test_nn_performances(x_train, x_test, y_train, y_test,
                            random_seed=random_seeds[0], max_iters=200,
                            init_temp=100, exp_decay_rate=0.1, min_temp=0.001,
                            pop_size=100, mutation_prob=0.2)


if __name__ == "__main__":

    random_seeds = [5 + 5 * i for i in range(2)]  # random seeds for get performances over multiple random runs

    # Define list of inter-distances between each pair of the following cities (in order from 0 to 9):
    # Rome, Florence, Barcelona, Paris, London, Amsterdam, Berlin, Prague, Budapest, Venice

    edges = [(1,2), (1,28), (2,3), (2,30), (3,30), (3,4), (4,5), (5,6), (5,27), (6,7), (6,11), (7,8),
            (7,11), (8,9), (9,10), (9,11), (10,12), (11,13), (12,13), (13,14), (14,16), (14,17), (15,16),
            (16,19), (16,20), (17,18), (17,27), (18,19), (18,26), (19,20), (20,21), (21,22), (22,23),
            (22,25), (23,24), (24,25), (24,28), (25,26), (26,27), (28,30)]

    # https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html
    coords = [(6.734, 1.453), (2.233, .010), (5.530, 1.424),
              (.401, .841), (3.082, 1.644), (7.608, 4.458),
              (7.573, 3.716), (7.265, 1.268), (6.898, 1.885),
              (1.112, 2.049), (5.468, 2.606), (5.989, 2.873)]
    # Experiment the Travel Salesman Problem, Flip Flop and Four Peaks with RHC, SA, GA and MIMIC
    travel_salesman(length=12, coords=coords, random_seeds=random_seeds)

    four_peaks(length=100, random_seeds=random_seeds)

    max_color(length=100, edges=edges, random_seeds=random_seeds)

    # queens(length=100, random_seeds=random_seeds)

    # Experiment Neural Networks optimization with RHC, SA, GA and GD on the WDBC dataset
    x_train, x_test, y_train, y_test = load_dataset(split_percentage=0.2, dataset1="diabetes_data_upload.csv")
    x_train1, x_test1, y_train1, y_test1 = load_dataset(split_percentage=0.2, data=load_wine())
    # x_train, x_test, y_train, y_test = load_dataset(split_percentage=0.2, data=load_wine())
    neural_network(x_train, x_test, y_train, y_test, random_seeds=random_seeds)
