import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob

# INPUT SUBMISSION
# 30.in : make_graph(28,3.69)
# 50.in : make_graph(48,2.119)
# 100.in: make_graph(100,0.963)

def make_graph(V, e):
    '''
    Args:
        V: # of vertices
        e: smallest edge length
    Returns:
        G: undirected graph with V vertices and min(edge length) = e
    '''
    def edge_maker(s,d):
        c1 = s
        for c2 in range(s+d,V,d):
            edges.append(str(c1) + ' ' + str(c2)  + ' ' + str(format(e*d, '.3f')))
            c1 = c2

    edges = []
    for i in range(V):
        for j in range(1,V):
            if i < j:
                edge_maker(i,j)

    G = nx.parse_edgelist(edges, nodetype=int, data=(("weight", float),))
    G.add_nodes_from(range(V))
    
    return G

def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """

    def get_removeables(A, c):
        """
        Args:
            A: list of cities/edges that can be removed
            c: # of cities/edges to remove
        Returns:
            R: list of nodes/edges to remove
        """
        if len(A) == 0 or c == 0 or len(A) < c:
            return [[]]
        R = []
        for i,n in enumerate(A):
            if len(A)-i >= c:
                for l in get_removeables(A[i+1:], c-1):
                    R.append([n] + l)
        return R


    def graph_generator(H, k, city):
        """
        Args:
            H: networkx.Graph
            k: # of cities/edges to remove
            city: variable to check whether to remove cities or edges
        Returns:
            L: list of all possible connected graphs w/ (H.nodes - k) cities if city is True 
               otherwise list of all possible connected graphs w/ (H.edges - k) edges
        """
        L = []
        if city:
            n = H.number_of_nodes()
            for nodes_to_remove in get_removeables(list(range(1,n)), k):
                h = H.copy()
                h.remove_nodes_from(nodes_to_remove)
                if nx.is_connected(h):
                    L.append(h) 
        else:
            E = list(H.edges)
            for edges_to_remove in get_removeables(E, k):
                h = H.copy()
                h.remove_edges_from(edges_to_remove)
                if nx.is_connected(h):
                    L.append(h)
        return L 

    # Initialize
    num_nodes,num_k,num_c = G.number_of_nodes(),0,0
    if num_nodes <= 30:
        num_k,num_c = 15,1
    elif num_nodes <= 50:
        num_k,num_c = 50,3
    elif num_nodes <= 100:
        num_k,num_c = 100,5

    answer = (G, nx.dijkstra_path_length(G,0,num_nodes-1))

    for cc in range(num_c):
        less_cities = graph_generator(G, cc, True)
        for g in less_cities:
            # Run Dijkstra's Algorithm on every possible route on g w/ up to num_k removed edges
            # copy graph g and remove edge(s). 
            # maybe it's a good idea to save graph or setting of max shortest path 
            for kk in range(1,num_k+1):
                less_edges = graph_generator(g, kk, False)
                for graph in less_edges:
                    p_len = nx.dijkstra_path_length(graph, 0, num_nodes-1)
                    if p_len > answer[1]:
                        answer = (graph, p_len)
    
    c = [v for v in G.nodes if v not in answer[0].nodes]
    k = [e for e in G.edges if e not in answer[0].edges]
    return c,k


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     c, k = solve(G)
#     assert is_valid_solution(G, c, k)
#     print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
#     write_output_file(G, c, k, 'outputs/small-1.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
input_path = 'inputs/inputs'
if __name__ == '__main__':
    inputs = glob.glob(input_path)
    for input_path in inputs:
        output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        distance = calculate_score(G, c, k)
        write_output_file(G, c, k, output_path)
