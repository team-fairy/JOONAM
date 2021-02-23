#!/usr/bin/env python
# coding: utf-8

# from google.colab import drive
# drive.mount('/content/drive')

import os
import os.path as osp
import time
from collections import defaultdict
from typing import DefaultDict, Dict, Set, TextIO, Tuple, Union, List
import numpy as np
import sys
from abc import *
np.set_printoptions(threshold=sys.maxsize)






# In[5]:


###################################################
#   Abstract Graph class                          #
#                                                 #
###################################################
class DiGraph(metaclass=ABCMeta):
    
    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def nodes(self) -> Set[int]:
        pass

    @abstractmethod
    def edges(self) -> List[Tuple[int, int]]:
        pass
    
    @abstractmethod
    def add_vertex(self, v: int) -> None:
        pass
    
    @abstractmethod
    def remove_vertex(self, v: int) -> None:
        pass

    @abstractmethod
    def add_edge(self, src: int, dst: int) -> None:
        pass

    @abstractmethod
    def remove_edge(self, src: int, dst: int) -> None:
        pass

    @abstractmethod
    def out_neighbor(self, v: int):
        pass

    @abstractmethod
    def out_degree(self, v: int) -> int:
        pass

    @abstractmethod
    def has_edge(self, src: int, dst: int) -> bool:
        pass


from queue import PriorityQueue
from typing import DefaultDict
from collections import defaultdict
import math

# Breath First Search algorithm
def shortest_path_from(G: DiGraph, src: int) -> defaultdict(int):
    queue = []
    visited = {v: False for v in G.nodes()}
    distance = {v: float('inf') for v in G.nodes()}
    visited[src] = True
    distance[src] = 0
    queue.append(src)
    
    while (len(queue) != 0):
        v = queue.pop(0)
        neighbors = G.out_neighbor(v)
        for u in G.out_neighbor(v):
            if visited[u] is False:
                visited[u] = True
                distance[u] = distance[v] + 1
                queue.append(u)
    return distance

# # **Homework 1-1**
# 
# TODO : Implement following methods for graph network                                 
# - edges()               : return list of edges [e1, e2, ...]               
# - add_vertex(v)         : add vertex v 
# - remove_vertex(v)      : remove vertex v 
# - add_edge(src, dst)    : add edge from one vertex(src) to another(dst)             
# - remove_edge(src, dst) : remove edge from src to dst                               
# - out_neighbor(v)       : get neighbors of vertex(v)                                
# - out_degree(v)         : get the number of edges from vertex v                     
# - has_edge(src, dst)    : return True if the edge (src, dst) is in the graph       



###################################################
#   Dictionary Array Graph                        #
#    - 정점       : self._nodes                   #             
#    - 나가는간선 : self._out_neighbor             #
#                                                 #
###################################################
class Graph_dict(DiGraph):

    def __init__(self) -> None:
        self._nodes = set()
        self._out_neighbor = defaultdict(set)

    def __str__(self):
        result = ""
        for v in self._nodes:
            neighbors = self._out_neighbor[v]
            if not neighbors:
                result = result + str(v) + " : empty\n"
            else : result = result + str(v) + " : " +  str(neighbors) + "\n"
        return result
    
    def nodes(self) -> Set[int]:
        return self._nodes

    def edges(self) -> List[Tuple[int, int]]:
        edges = []
        for v1 in self._nodes:
            neighbors = self._out_neighbor[v1]
            for v2 in neighbors:
                edges.append((v1, v2))
        return edges
        


    ################# Write your answer here #####################
    def add_vertex(self, v: int) -> None:
        self._nodes.add(v)

    ################# Write your answer here #####################
    def remove_vertex(self, v: int) -> None:
        self._nodes.remove(v)
        for i in self._out_neighbor:
            self.remove_edge(i,v)
            self.remove_edge(v,i)

    def add_edge(self, src: int, dst: int) -> None:
        self.add_vertex(src)
        self.add_vertex(dst)
        self.out_neighbor(src).add(dst)

    def remove_edge(self, src: int, dst: int) -> None:
        neighbor = self.out_neighbor(src)
        if dst in neighbor:
            self._out_neighbor[src].remove(dst)

    ################# Write your answer here #####################
    def out_neighbor(self, v: int) -> Set[int]:
        return self._out_neighbor[v]
    
    ################# Write your answer here #####################
    def out_degree(self, v: int) -> int:
        # raise NotImplemented
        return len(self._out_neighbor[v])
    
    def has_edge(self, src: int, dst: int) -> bool:
        return dst in self._out_neighbor[src]
    
    ################# Write your answer here #####################
    # HW 1-2
    def subgraph(self, vertices: List[int]):
        subgraph = Graph_dict()
        for n in vertices:
            subgraph.add_vertex(n)
        for src in vertices:
            for dst in self._out_neighbor[src]:
                if dst in vertices:
                    subgraph.add_edge(src, dst)          
            
        return subgraph



# In[7]:


###################################################
#   Numpy Array Graph                             #
#    - 정점     : self._nodes                     #
#    - 간선     : self._data                      #
#    - max_size : self._max_size                  #
#                                                 #
###################################################
class Graph_numpy_array(DiGraph):
    def __init__(self, size: int) -> None:
        self._data = np.zeros((size, size))
        self._max_size = size
        self._nodes = set()

    def __str__(self):
        return str(self._data)
    
    def nodes(self) -> Set[int]:
        return self._nodes

    def edges(self) -> List[Tuple[int, int]]:
        edge_list = []
        for i in range(self._max_size):
            for j in range(self._max_size):
                if self._data[i][j] == 1:
                    edge_list.append((i, j))
        return edge_list

    def add_vertex(self, v: int) -> None:
        if v >= 0 and v < self._max_size:
            self._nodes.add(v)
        else:
            print("Error : vertex number is in the out of range")
    
    def remove_vertex(self, v: int) -> None:
        if v >= 0 and v < self._max_size:
            self._nodes.remove(v)
            self._data[v, :] = 0
            self._data[:, v] = 0
        else :
            print("Error : vertex number is in the out of range")

    ################# Write your answer here #####################
    def add_edge(self, src: int, dst: int) -> None:
        # raise NotImplemented
        self.add_vertex(src)
        self.add_vertex(dst)
        self._data[src,dst] = 1
        # self._data[dst,src] = 1

    ################# Write your answer here #####################
    def remove_edge(self, src: int, dst: int) -> None:
        # raise NotImplemented
        self._data[src,dst] = 0
        # self._data[dst,src] = 0


    def out_neighbor(self, v: int) -> Set[int]:
        return {x for x in range(self._max_size) if self._data[v][x] == 1}

    def out_degree(self, v: int) -> int:
        return len(self.out_neighbor(v))
    
    ################# Write your answer here #####################
    def has_edge(self, src: int, dst: int) -> bool:
        res = False
        if self._data[src,dst] == 1:
            res = True
        return res

    def subgraph(self, vertices: List[int]):
        subgraph = Graph_numpy_array(self._max_size)
        for v in vertices:
            subgraph.add_vertex(v)
        for v1 in vertices:
            for v2 in vertices:
                if v1 == v2:
                    continue
                elif self.has_edge(v1, v2):
                    subgraph.add_edge(v1, v2)
        return subgraph





# Test 1 : subgraph
if __name__ == '__main__':

    SUBMISSION = False
    if not(SUBMISSION):
        root_path = '/content/drive/MyDrive/WorkColab/AI_Boostcamp/week5_graph'
        root_path = './'
        os.chdir(root_path)
    import networkx as nx

    G = Graph_dict()
    G2 = Graph_numpy_array(100)
    for i in range(0, 10):
        G.add_edge(i, i+1)
        G2.add_edge(i, i+1)

    H = G.subgraph([1, 2, 3, 4, 5])
    H2 = G2.subgraph([1, 2, 3, 4, 5])
    assert H.edges() == [(1, 2), (2, 3), (3, 4), (4, 5)]
    assert H.edges() == H2.edges()

    H = G.subgraph([1, 3, 5])
    H2 = G2.subgraph([1, 3, 5])
    assert H.edges() == []
    assert H.edges() == H2.edges()

    H = G.subgraph(G.nodes())
    H2 = G2.subgraph(G.nodes())
    assert H.edges() == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
    assert H.edges() == H2.edges()
    assert H.edges() == G.edges()


