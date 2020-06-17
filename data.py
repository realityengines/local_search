import numpy as np
import pickle
import sys
import os

from nas_bench.cell import Cell
from darts.arch import Arch



class Data:

    def __init__(self, search_space, mf=False, dataset='cifar10', nasbench_folder='./', loaded_nasbench=None):
        self.search_space = search_space
        self.mf = mf
        self.dataset = dataset

        if search_space == 'nasbench':
            if loaded_nasbench:
                self.nasbench = loaded_nasbench
            else:
                from nasbench import api

                if mf:
                    self.nasbench = api.NASBench(nasbench_folder + 'nasbench_full.tfrecord')
                else:
                    self.nasbench = api.NASBench(nasbench_folder + 'nasbench_only108.tfrecord')

        elif search_space == 'darts':
            from darts.arch import Arch
        else:
            print(search_space, 'is not a valid search space')
            sys.exit()

    def get_type(self):
        return self.search_space

    def get_mf(self):
        return self.mf

    def epoch_encoding(self, encoding, epochs, change=False):
        """
        Add or change the encoding of an arch to a fidelity (epochs).
        Currently only set up for nasbench space.
        """
        if change:
            encoding = encoding[:-4]

        if epochs == 4:
            encoding = [*encoding, *[1,0,0,0]]
        elif epochs == 12:
            encoding = [*encoding, *[0,1,0,0]]
        elif epochs == 36:
            encoding = [*encoding, *[0,0,1,0]]
        else:
            encoding = [*encoding, *[0,0,0,1]]
        return encoding

    def convert_to_cells(self, 
                            arches, 
                            encoding_type='path',
                            cutoff=40,
                            train=True):
        cells = []
        for arch in arches:
            spec = Cell.convert_to_cell(arch)
            cell = self.query_arch(spec,
                                   encoding_type=encoding_type,
                                   cutoff=cutoff,
                                   train=train)
            cells.append(cell)

        return cells

    def query_arch(self, 
                   arch=None, 
                   train=True, 
                   encoding_type='adj', 
                   random='standard',
                   deterministic=True, 
                   epochs=0,
                   cutoff=-1,
                   random_hash=False,
                   max_edges=-1,
                   max_nodes=-1):

        arch_dict = {}
        arch_dict['epochs'] = epochs
        if self.search_space in ['nasbench', 'nasbench_201']:
            if arch is None:
                if max_edges > 0 or max_nodes > 0:
                    arch = Cell.random_cell_constrained(self.nasbench, 
                                                        max_edges=max_edges,
                                                        max_nodes=max_nodes)
                elif encoding_type == 'continuous':
                    arch = Cell.random_cell_continuous(self.nasbench)
                elif random == 'uniform':
                    arch = Cell.random_cell_uniform(self.nasbench)
                else:
                    arch = Cell.random_cell(self.nasbench)
            arch_dict['spec'] = arch

            if encoding_type == 'path':
                encoding = Cell(**arch).encode_paths()
            elif encoding_type == 'path-short':
                encoding = Cell(**arch).encode_paths()[:40]
            elif encoding_type == 'continuous':
                encoding = Cell(**arch).encode_continuous()
            elif encoding_type == 'freq':
                encoding = Cell(**arch).encode_freq_paths(cutoff=cutoff)
            else:
                encoding = Cell(**arch).encode_standard()

            arch_dict['encoding'] = self.epoch_encoding(encoding, epochs)

            # special keys for local search and outside_ss experiments
            if self.search_space == 'nasbench_201' and random_hash:
                arch_dict['random_hash'] = Cell(**arch).get_random_hash()
            if self.search_space == 'nasbench':
                arch_dict['adjacency'] = Cell(**arch).encode_standard()
                arch_dict['path'] = Cell(**arch).encode_paths()

            if train:
                if not self.get_mf():
                    arch_dict['val_loss'] = Cell(**arch).get_val_loss(self.nasbench, 
                                                                        deterministic=deterministic,
                                                                        dataset=self.dataset)
                    arch_dict['test_loss'] = Cell(**arch).get_test_loss(self.nasbench,
                                                                        dataset=self.dataset)
                else:
                    arch_dict['val_loss'] = Cell(**arch).get_val_loss(self.nasbench, 
                                                                        deterministic=deterministic, 
                                                                        epochs=epochs)
                    arch_dict['test_loss'] = Cell(**arch).get_test_loss(self.nasbench, epochs=epochs)

                arch_dict['num_params'] = Cell(**arch).get_num_params(self.nasbench)
                arch_dict['val_per_param'] = (arch_dict['val_loss'] - 4.8) * (arch_dict['num_params'] ** 0.5) / 100

                if self.search_space == 'nasbench':
                    arch_dict['dist_to_min'] = arch_dict['val_loss'] - 4.94457682
                elif self.dataset == 'cifar10':
                    arch_dict['dist_to_min'] = arch_dict['val_loss'] - 8.3933
                elif self.dataset == 'cifar100':
                    arch_dict['dist_to_min'] = arch_dict['val_loss'] - 26.5067
                else:
                    arch_dict['dist_to_min'] = arch_dict['val_loss'] - 53.2333


        else:
            if arch is None:
                arch = Arch.random_arch()

            if encoding_type == 'path':
                encoding = Arch(arch).encode_paths()
            elif encoding_type == 'path-short':
                encoding = Arch(arch).encode_freq_paths()
            else:
                encoding = arch
            arch_dict['spec'] = arch

            # todo add mf encoding options here
            arch_dict['encoding'] = encoding

            if train:
                if epochs == 0:
                    epochs = 50
                arch_dict['val_loss'], arch_dict['test_loss'] = Arch(arch).query(epochs=epochs)
        
        return arch_dict           

    def mutate_arches(self, arches):
        # method for metann_outside. currently not being used
        mutated = []
        for arch in arches:
            for _ in range(10):
                for e in range(1, 11):
                    mutated = mutate_arch(arch, mutation_rate=e)
                    mutations.append(mutated)

        return mutations    

    def mutate_arch(self, arch, 
                    mutation_rate=1.0, 
                    encoding_type='adjacency', 
                    comparisons=2500,
                    cutoff=-1):
        if self.search_space in ['nasbench', 'nasbench_201']:
            return Cell(**arch).mutate(self.nasbench, 
                                        mutation_rate=mutation_rate, 
                                        encoding_type=encoding_type,
                                        comparisons=comparisons,
                                        cutoff=cutoff)
        else:
            return Arch(arch).mutate(int(mutation_rate))

    def get_nbhd(self, arch, nbhd_type='full'):
        if self.search_space in ['nasbench', 'nasbench_201']:
            return Cell(**arch).get_neighborhood(self.nasbench, nbhd_type=nbhd_type)
        else:
            return Arch(arch).get_neighborhood(nbhd_type=nbhd_type)

    def get_hash(self, arch, epochs=0):
        # return a unique hash of the architecture+fidelity
        # we use path indices + epochs
        if self.search_space == 'nasbench':
            return (*Cell(**arch).get_path_indices(), epochs)
        elif self.search_space == 'darts':
            return (*Arch(arch).get_path_indices()[0], epochs)
        else:
            return Cell(**arch).get_string()

    # todo change kwarg to deterministic
    def generate_random_dataset(self,
                                num=10, 
                                train=True,
                                encoding_type='adj', 
                                random='standard',
                                allow_isomorphisms=False, 
                                deterministic_loss=True,
                                patience_factor=5,
                                mf_type=None,
                                cutoff=-1,
                                max_edges=-1,
                                max_nodes=-1):
        """
        create a dataset of randomly sampled architectues
        test for isomorphisms using a hash map of path indices
        use patience_factor to avoid infinite loops
        """
        data = []
        dic = {}
        tries_left = num * patience_factor
        while len(data) < num:
            tries_left -= 1
            if tries_left <= 0:
                break
            epochs = 0
            if mf_type:
                epochs = sample_fidelity(mf_type, query_proportion=0)

            arch_dict = self.query_arch(train=train,
                                        encoding_type=encoding_type,
                                        random='standard',
                                        deterministic=deterministic_loss,
                                        epochs=epochs,
                                        cutoff=cutoff,
                                        max_edges=max_edges,
                                        max_nodes=max_nodes)

            h = self.get_hash(arch_dict['spec'], epochs)

            if allow_isomorphisms or h not in dic:
                dic[h] = 1
                data.append(arch_dict)

        return data

