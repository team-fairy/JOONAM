import os
import os.path as osp
import time
from collections import defaultdict
from typing import DefaultDict, Dict, Set, TextIO, Tuple, Union, List
import numpy as np
import sys
from abc import *
from subgraph import Graph_dict
from subgraph import Graph_numpy_array
np.set_printoptions(threshold=sys.maxsize)

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

# 
# # **Homework 1-2**
#                                                                     
# TODO : Implement following methods for pagerank algorithm                           
# 1) Graph_dict   
# First, you implement subgraph methods in the graph_dict class 
# 
# - subgraph(vertices) : get the induced subgraph containing the nodes and edges
# 
# 2) pagerank(graph, damping_factor, maxiters, tol)   
# Compute the distance between two dictionaries based on L1 norm
def l1_distance(x: DefaultDict[int, float], y: DefaultDict[int, float]) -> float:
    err: float = 0.0
    for k in x.keys():
        err += abs(x[k] - y[k])
    return err


################################################################################
# Run the pagerank algorithm iteratively using the graph_dict                  #
#  parameters                                                                  #
#    - graph : Directed graph (graph_dict object)                              #
#    - damping_factor : Damping factor                                         #
#    - maxiters : The maximum number of iterations                             #
#    - tol : Tolerance threshold to check the convergence                      #
################################################################################


def pagerank(
    graph: Graph_dict,
    damping_factor: float,
    maxiters: int,
    tol: float,
) -> Dict[int, float]:
    vec: DefaultDict[int, float] = defaultdict(float)  # Pagerank vector
    

    ############### TODO: Implement the pagerank algorithm #####################    
    V = graph.nodes()
    E = graph.edges()

    vec = defaultdict(float)
    for v in V:
        vec[v] = 1/len(V)

    for itr in range(maxiters):
        vec_new = defaultdict(float)
        score = 0
        for j in V:
            vec_new[j] = 0
            for i in set([ edge[0] for edge in E if j == edge[1] ]):
                vec_new[j] += damping_factor*( vec[i] / graph.out_degree(i) )
            score += vec_new[j]
        for j in V:
            vec_new[j] += (1-score)/len(V)

        #### Check the convergence ###

        # Stop the iteration if L1norm[PR(t) - PR(t-1)] < tol
        delta: float = 0.0
        delta = l1_distance(vec_new, vec)

        print(f"[Iter {itr}]\tDelta = {delta}")

        if delta < tol:
            break
        
        vec = vec_new
        del vec_new

    ########################### Implementation end #############################

    return dict(vec)


# You can test your pagerank algorithm using following code. (same as lab3)




# In[11]:
G = Graph_dict()
if SUBMISSION : 
    path_v2n = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/others/vertex2name.txt'))
    path_edges = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/others/edges.txt'))

    # keyword : deep_learning.txt (딥러닝), lee.txt (이순신), bong.txt(봉준호)
    path_keyword = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/hw/hw1/deep_learning.txt'))
else:
    path_v2n = "data/others/vertex2name.txt"
    path_edges = "data/others/edges.txt"
    path_keyword = "data/hw/hw1/deep_learning.txt"


f = open(path_edges)
for line in f:
    v1, v2 = map(int, line.split())
    G.add_edge(v1, v2)                                                 
                                 
n2v = {}
v2n = {}
f = open(path_v2n, encoding='UTF8')
for line in f:
    v, n = line.split()
    v = int(v)
    n = n.rstrip()
    n2v[n] = v
    v2n[v] = n

node_key = []
f = open(path_keyword)
for line in f:
    v = line.rstrip()
    v = int(v)
    node_key.append(v)

H = G.subgraph(node_key)

print("###### PageRank Algorithm ######")
pr = pagerank(H, damping_factor = 0.9 , maxiters = 100, tol = 1e-06)
res = [key for (key, value) in sorted(pr.items(), key=lambda x:x[1], reverse=True)]
res_value = [value for (key, value) in sorted(pr.items(), key=lambda x:x[1], reverse=True)]
for item in res[:10]:
    print(v2n[item])


''' 
Expected Output (solution) 

###### PageRank Algorithm ######
[Iter 0]	Delta = 0.2556213017751478
[Iter 1]	Delta = 0.03539371870732816
[Iter 2]	Delta = 0.004900668744091563
[Iter 3]	Delta = 0.0006785541337973044
[Iter 4]	Delta = 9.395364929493111e-05
[Iter 5]	Delta = 1.3008966825434065e-05
[Iter 6]	Delta = 1.8012415605173082e-06
[Iter 7]	Delta = 2.494026775656799e-07
딥러닝
PyTorch
OpenCV
라온피플
이스트소프트
인공지능인문학
자동긴급제동장치
케플러-90i
T2d
심심이

'''





# Test 1 : subgraph
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