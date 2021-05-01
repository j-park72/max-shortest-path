import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob


# The main idea here is to come up with a shortest-path-tree 
# by removing edges/cities from parent graph's shortest path.
# Use Heuristic, HTC which determines how deep I go down the tree 
# since going all the way down the tree takes up too much resources.
# e.g. height of tree = 5, HTC = 2:
# I go down the tree 2 height then amongst the graphs I have so far, I find the one w/ max shortest path. 
# Then go 2 more down the tree and so on until I reach the bottom of the tree. 
# Then compare and go w/ max shortest path.

def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """

    # Heuristic: height of a shortest-path-tree to merge
    HTC = 3

    def nodes_to_edges(nodes):
        """
        Args:
            nodes: list of nodes to convert
        Returns:
            L: converted list of edges
        """ 
        edges = []
        if len(nodes) == 0:
            return edges
        prev = nodes[0]
        for n in nodes[1:]:
            edges.append((prev, n))
            prev = n
        return edges

    def solver(A, k, is_edges):
        """
        Args:
            A: tuple of (G, shortest-path-length of G, shortest-path of G)
            k: # of nodes/edges to remove (i.e. total height of tree)
            is_edges: determines whether solving for cities or edges
        Returns:
            A: tuple of G w/ k removed nodes/edges that has the max-shortest-path
        """
        def down_edge_tree(A, k):
            if k == 0:
                return [A]
            R = []
            for e in A[2]:
                H = A[0].copy()
                H.remove_edge(e[0], e[1])
                if nx.is_connected(H):
                    shortest_path = nx.single_source_dijkstra(H,0,dest)
                    B = (H,shortest_path[0],nodes_to_edges(shortest_path[1]))
                    for x in down_edge_tree(B, k-1):
                        R.append(x)
            return R

        def down_node_tree(A, k):
            if k == 0:
                return [A]
            R = []
            for n in A[2]:
                H = A[0].copy()
                H.remove_node(n)
                if nx.is_connected(H):
                    shortest_path = nx.single_source_dijkstra(H,0,dest)
                    B = (H, shortest_path[0], shortest_path[1])
                    for x in down_node_tree(B, k-1):
                        R.append(x)
            return R
        
        recurser = down_edge_tree if is_edges else down_node_tree

        while k > 0:
            R = recurser(A, k) if k < HTC else recurser(A, HTC)
            r = list(map(lambda x: x[1], R))
            if len(r) == 0:
                break
            A = R[r.index(max(r))]
            k -= HTC

        return A

    # Initialize
    num_k, num_c, dest = 0, 0, G.number_of_nodes()-1
    if G.number_of_nodes() <= 30:
        num_k, num_c = 15, 1
    elif G.number_of_nodes() <= 50:
        num_k, num_c = 50, 3
    elif G.number_of_nodes() <= 100:
        num_k, num_c = 100, 5

    # Remove edges
    dijkstra = nx.single_source_dijkstra(G,0,dest)
    answer = (G, dijkstra[0],nodes_to_edges(dijkstra[1]))
    A = solver(answer[0], num_k, True)
    if answer[1] < A[1]:
        answer = A
    k = [e for e in G.edges if e not in answer[0].edges]

    # Remove cities
    A = solver(answer[0], num_c, False)
    if answer[1] < A[1]:
        answer = A
    c = [v for v in G.nodes if v not in answer[0].nodes]

    return c, k





# Here's an example of how to run your solver.
set_type = 'medium'
input_path = 'inputs/' + set_type + '/'

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     filename = sys.argv[1]
#     G = read_input_file(input_path + filename)
#     c, k = solve(G)
#     assert is_valid_solution(G, c, k)
#     print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
#     write_output_file(G, c, k, 'singleOutputs/' + set_type + '/' + filename[:-3] + '.out')



# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)

# if __name__ == '__main__':
#     inputs = glob.glob(input_path + '*')
#     for input_path in inputs:
#         output_path = 'outputs/' + set_type + '/' + \
#             basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         write_output_file(G, c, k, output_path)
