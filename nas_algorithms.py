import itertools
import os
import pickle
import sys
import copy
import numpy as np
import tensorflow as tf
from argparse import Namespace

from data import Data

def run_nas_algorithm(algo_params, search_space, mp):

    # run nas algorithm
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')

    if algo_name == 'random':
        data = random_search(search_space, **ps)
    elif algo_name == 'evolution':
        data = evolution_search(search_space, **ps)
    elif algo_name == 'local_search':
        data = local_search(search_space, **ps)

    else:
        print('invalid algorithm name')
        sys.exit()

    k = 10
    if 'k' in ps:
        k = ps['k']
    total_queries = 150
    if 'total_queries' in ps:
        total_queries = ps['total_queries']
    loss = 'val_loss'
    if 'loss' in ps:
        loss = ps['loss']

    return compute_best_test_losses(data, k, total_queries, loss), data


# todo
# this method is only for single fidelity
# make a new file for evaluating by runtime
def compute_best_test_losses(data, k, total_queries, loss):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error 
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i:i[loss])[0]
        test_error = best_arch['test_loss']
        results.append((query, test_error))

    return results


def random_search(search_space,
                    total_queries=150, 
                    loss='val_loss',
                    deterministic=True,
                    verbose=1):
    """ 
    random search
    """
    data = search_space.generate_random_dataset(num=total_queries, 
                                                deterministic_loss=deterministic)
    
    if verbose:
        top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
        print('random, query {}, top 5 losses {}'.format(total_queries, top_5_loss))    
    return data


def evolution_search(search_space,
                        num_init=10,
                        k=10,
                        loss='val_loss',
                        population_size=30,
                        total_queries=150,
                        tournament_size=10,
                        mutation_rate=1.0, 
                        comparisons=2500,
                        deterministic=True,
                        regularize=True,
                        verbose=1):
    """
    regularized evolution
    """
    data = search_space.generate_random_dataset(num=num_init, 
                                                deterministic_loss=deterministic)

    losses = [d[loss] for d in data]
    query = num_init

    population = [i for i in range(min(num_init, population_size))]

    while query <= total_queries:

        # evolve the population by mutating the best architecture
        # from a random subset of the population
        sample = np.random.choice(population, tournament_size)
        best_index = sorted([(i, losses[i]) for i in sample], key=lambda i:i[1])[0][0]
        mutated = search_space.mutate_arch(data[best_index]['spec'], 
                                            mutation_rate=mutation_rate, 
                                            comparisons=comparisons)
        arch_dict = search_space.query_arch(mutated, deterministic=deterministic)

        data.append(arch_dict)        
        losses.append(arch_dict[loss])
        population.append(len(data) - 1)

        # kill the oldest (or worst) from the population
        if len(population) >= population_size:
            if regularize:
                oldest_index = sorted([i for i in population])[0]
                population.remove(oldest_index)
            else:
                worst_index = sorted([(i, losses[i]) for i in population], key=lambda i:i[1])[-1][0]
                population.remove(worst_index)

        if verbose and (query % k == 0):
            top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
            print('evolution, query {}, top 5 losses {}'.format(query, top_5_loss))

        query += 1

    return data


def local_search(search_space,
                num_init=10,
                k=10,
                loss='val_loss',
                nbhd_type_pattern=['full'],
                query_full_nbhd=False,
                stop_at_minimum=False,
                total_queries=500,
                deterministic=True,
                verbose=1):
    """
    local search
    """

    query_dict = {}
    iter_dict = {}
    data = []
    query = 0

    while True:
        # loop over full runs of local search until queries run out
        
        arch_dicts = []
        while len(arch_dicts) < num_init:
            arch_dict = search_space.query_arch(deterministic=deterministic)

            if search_space.get_hash(arch_dict['spec']) not in query_dict:
                query_dict[search_space.get_hash(arch_dict['spec'])] = 1
                data.append(arch_dict)
                arch_dicts.append(arch_dict)
                query += 1
                if query >= total_queries:
                    return data

        sorted_arches = sorted([(arch, arch[loss]) for arch in arch_dicts], key=lambda i:i[1])
        arch_dict = sorted_arches[0][0]                
        nbhd_type_num = -1

        while True:
            # loop over iterations of local search until we hit a local minimum
            if verbose:
                print('starting iteration, query', query)
            nbhd_type_num = (nbhd_type_num + 1) % len(nbhd_type_pattern)
            iter_dict[search_space.get_hash(arch_dict['spec'])] = 1
            nbhd = search_space.get_nbhd(arch_dict['spec'], 
                                        nbhd_type=nbhd_type_pattern[nbhd_type_num])
            improvement = False
            nbhd_dicts = []
            for nbr in nbhd:
                if search_space.get_hash(nbr) not in query_dict:
                    query_dict[search_space.get_hash(nbr)] = 1

                    nbr_dict = search_space.query_arch(nbr, deterministic=deterministic)
                    data.append(nbr_dict)
                    nbhd_dicts.append(nbr_dict)
                    query += 1
                    if query >= total_queries:
                        return data
                    if nbr_dict[loss] < arch_dict[loss]:
                        improvement = True
                        if not query_full_nbhd:
                            arch_dict = nbr_dict
                            break

            if not stop_at_minimum:
                sorted_data = sorted([(arch, arch[loss]) for arch in data], key=lambda i:i[1])
                index = 0
                while search_space.get_hash(sorted_data[index][0]['spec']) in iter_dict:
                    index += 1

                arch_dict = sorted_data[index][0]

            elif not improvement:
                break

            else:
                sorted_nbhd = sorted([(nbr, nbr[loss]) for nbr in nbhd_dicts], key=lambda i:i[1])
                arch_dict = sorted_nbhd[0][0]

            # check for repeats. remove this later if it's working consistently
            dic = {}
            count = 0
            for d in data:
                if search_space.get_hash(d['spec']) not in dic:
                    dic[search_space.get_hash(d['spec'])] = 1
                else:
                    count += 1
            if count:
                print('there were {} repeats'.format(count))


        if verbose:
            top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
            print('local_search, query {}, top 5 losses {}'.format(query, top_5_loss))



