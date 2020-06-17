import numpy as np
import copy
import itertools
import random
import sys
import os

from nasbench import api


INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class Cell:

    def __init__(self, matrix, ops):

        self.matrix = matrix
        self.ops = ops

    def serialize(self):
        return {
            'matrix': self.matrix,
            'ops': self.ops
        }


    def get_utilized(self):
        # return the sets of utilized edges and nodes
        # first, compute all paths
        n = np.shape(self.matrix)[0]
        sub_paths = []
        for j in range(0, n):
            sub_paths.append([[(0, j)]]) if self.matrix[0][j] else sub_paths.append([])
        
        # create paths sequentially
        for i in range(1, n - 1):
            for j in range(1, n):
                if self.matrix[i][j]:
                    for sub_path in sub_paths[i]:
                        sub_paths[j].append([*sub_path, (i, j)])
        paths = sub_paths[-1]

        utilized_edges = []
        for path in paths:
            for edge in path:
                if edge not in utilized_edges:
                    utilized_edges.append(edge)

        utilized_nodes = []
        for i in range(NUM_VERTICES):
            for edge in utilized_edges:
                if i in edge and i not in utilized_nodes:
                    utilized_nodes.append(i)

        return utilized_edges, utilized_nodes

    def num_edges_and_vertices(self):
        # return the true number of edges and vertices
        edges, nodes = self.get_utilized()
        return len(edges), len(nodes)

    def is_valid_vertex(self, vertex):
        edges, nodes = self.get_utilized()
        return (vertex in nodes)

    def is_valid_edge(self, edge):
        edges, nodes = self.get_utilized()
        return (edge in edges)

    def modelspec(self):
        return api.ModelSpec(matrix=self.matrix, ops=self.ops)

    @classmethod
    def random_cell(cls, nasbench):
        """ 
        From the NASBench repository 
        https://github.com/google-research/nasbench
        """
        while True:
            matrix = np.random.choice(
                [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if nasbench.is_valid(spec):
                return {
                    'matrix': matrix,
                    'ops': ops
                }

    @classmethod
    def convert_to_cell(cls, arch):
        matrix, ops = arch['matrix'], arch['ops']

        if len(matrix) < 7:
            # the nasbench spec can have an adjacency matrix of n x n for n<7, 
            # but in the nasbench api, it is always 7x7 (possibly containing blank rows)
            # so this method will add a blank row/column

            new_matrix = np.zeros((7, 7), dtype='int8')
            new_ops = []
            n = matrix.shape[0]
            for i in range(7):
                for j in range(7):
                    if j < n - 1 and i < n:
                        new_matrix[i][j] = matrix[i][j]
                    elif j == n - 1 and i < n:
                        new_matrix[i][-1] = matrix[i][j]

            for i in range(7):
                if i < n - 1:
                    new_ops.append(ops[i])
                elif i < 6:
                    new_ops.append('conv3x3-bn-relu')
                else:
                    new_ops.append('output')
            return {
                'matrix': new_matrix,
                'ops': new_ops
            }

        else:
            return {
                'matrix': matrix,
                'ops': ops
            }

    def get_val_loss(self, nasbench, deterministic=1, patience=50, epochs=None, dataset=None):
        if not deterministic:
            # output one of the three validation accuracies at random
            if epochs:
                return (100*(1 - nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops), epochs=epochs)['validation_accuracy']))
            else:
                return (100*(1 - nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']))
        else:        
            # query the api until we see all three accuracies, then average them
            # a few architectures only have two accuracies, so we use patience to avoid an infinite loop
            accs = []
            while len(accs) < 3 and patience > 0:
                patience -= 1
                if epochs:
                    acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops), epochs=epochs)['validation_accuracy']
                else:
                    acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
                if acc not in accs:
                    accs.append(acc)
            return round(100*(1-np.mean(accs)), 4)            


    def get_test_loss(self, nasbench, patience=50, epochs=None, dataset=None):
        """
        query the api until we see all three accuracies, then average them
        a few architectures only have two accuracies, so we use patience to avoid an infinite loop
        """
        accs = []
        while len(accs) < 3 and patience > 0:
            patience -= 1
            if epochs:
                acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops), epochs=epochs)['test_accuracy']
            else:
                acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['test_accuracy']
            if acc not in accs:
                accs.append(acc)
        return round(100*(1-np.mean(accs)), 4)

    def get_num_params(self, nasbench):
        return nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['trainable_parameters']

    def perturb(self, nasbench, edits=1):
        """ 
        create new perturbed cell 
        inspird by https://github.com/google-research/nasbench
        """
        new_matrix = copy.deepcopy(self.matrix)
        new_ops = copy.deepcopy(self.ops)
        for _ in range(edits):
            while True:
                if np.random.random() < 0.5:
                    for src in range(0, NUM_VERTICES - 1):
                        for dst in range(src+1, NUM_VERTICES):
                            new_matrix[src][dst] = 1 - new_matrix[src][dst]
                else:
                    for ind in range(1, NUM_VERTICES - 1):
                        available = [op for op in OPS if op != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)

                new_spec = api.ModelSpec(new_matrix, new_ops)
                if nasbench.is_valid(new_spec):
                    break
        return {
            'matrix': new_matrix,
            'ops': new_ops
        }

    def mutate(self, nasbench, 
                mutation_rate=1.0, 
                encoding_type='adjacency', 
                cutoff=40, 
                comparisons=2500,
                patience=5000):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        p = 0
        while p < patience:
            p += 1
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            edge_mutation_prob = mutation_rate / NUM_VERTICES
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = mutation_rate / OP_SPOTS
            for ind in range(1, OP_SPOTS + 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in OPS if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops
                }
        return self.mutate(nasbench, mutation_rate+1, encoding_type=encoding_type)


    def encode_standard(self):
        """ 
        compute the "standard" encoding,
        i.e. adjacency matrix + op list encoding 
        """
        encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS
        encoding = np.zeros((encoding_length))
        dic = {CONV1X1: 0., CONV3X3: 0.5, MAXPOOL3X3: 1.0}
        n = 0
        for i in range(NUM_VERTICES - 1):
            for j in range(i+1, NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, NUM_VERTICES - 1):
            encoding[-i] = dic[self.ops[i]]
        return tuple(encoding)


    def edit_distance(self, other):
        """
        compute the distance between two architectures
        by comparing their adjacency matrices and op lists
        """
        graph_dist = np.sum(np.array(self.matrix) != np.array(other.matrix))
        ops_dist = np.sum(np.array(self.ops) != np.array(other.ops))
        return (graph_dist + ops_dist)

    def get_neighborhood(self, nasbench, nbhd_type='op', shuffle=True):
        nbhd = []
        if nbhd_type in ['op', 'full']:
            for vertex in range(1, OP_SPOTS + 1):
                if self.is_valid_vertex(vertex):
                    available = [op for op in OPS if op != self.ops[vertex]]
                    for op in available:
                        new_matrix = copy.deepcopy(self.matrix)
                        new_ops = copy.deepcopy(self.ops)
                        new_ops[vertex] = op
                        new_arch = {'matrix':new_matrix, 'ops':new_ops}
                        nbhd.append(new_arch)

        if nbhd_type != 'op':
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src+1, NUM_VERTICES):
                    new_matrix = copy.deepcopy(self.matrix)
                    new_ops = copy.deepcopy(self.ops)
                    new_matrix[src][dst] = 1 - new_matrix[src][dst]
                    new_arch = {'matrix':new_matrix, 'ops':new_ops}
                
                    if self.matrix[src][dst] and nbhd_type != 'add_edge' \
                    and self.is_valid_edge((src, dst)):
                        spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
                        if nasbench.is_valid(spec):                            
                            nbhd.append(new_arch)  

                    if not self.matrix[src][dst] and nbhd_type != 'sub_edge' \
                    and Cell(**new_arch).is_valid_edge((src, dst)):
                        spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
                        if nasbench.is_valid(spec):                            
                            nbhd.append(new_arch)                          
  
        if shuffle:
            random.shuffle(nbhd)
        return nbhd


    def get_paths(self):
        """ 
        return all paths from input to output
        """
        paths = []
        for j in range(0, NUM_VERTICES):
            paths.append([[]]) if self.matrix[0][j] else paths.append([])
        
        # create paths sequentially
        for i in range(1, NUM_VERTICES - 1):
            for j in range(1, NUM_VERTICES):
                if self.matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, self.ops[i]])
        return paths[-1]


    def get_path_indices(self):
        """
        compute the index of each path
        There are 3^0 + ... + 3^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are three choices for the operation.)
        """
        paths = self.get_paths()
        mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
        path_indices = []

        for path in paths:
            index = 0
            for i in range(NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(OPS) ** i * (mapping[path[i]] + 1)

        path_indices.sort()
        return tuple(path_indices)

    def encode_paths(self):
        """ output one-hot encoding of paths """
        num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
        path_indices = self.get_path_indices()
        encoding = np.zeros(num_paths)
        for index in path_indices:
            encoding[index] = 1
        return encoding
