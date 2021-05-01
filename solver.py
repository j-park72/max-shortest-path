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
    def edge_maker(s, d):
        c1 = s
        for c2 in range(s+d, V, d):
            edges.append(str(c1) + ' ' + str(c2) +
                         ' ' + str(format(e*d, '.3f')))
            c1 = c2

    edges = []
    for i in range(V):
        for j in range(1, V):
            if i < j:
                edge_maker(i, j)

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

    # Heuristic - height of a shortest-path-tree
    HTC = 3

    def graph_generator(H):
        """
        Args:
            H: networkx.Graph
            k: # of cities to remove
        Returns:
            L: list of all possible connected graphs w/ (H.nodes - 1) cities 
        """
        L = []
        nodes = list(H.nodes)
        nodes.remove(0)
        nodes.remove(dest)
        for node in nodes:
            h = H.copy()
            h.remove_node(node)
            if nx.is_connected(h):
                L.append(h)

        return L

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

    def solver(A, k):
        """
        Args:
            A: 
        Returns:
            L: converted list of edges
        """
        def helper(A, k):
            if k == 0:
                return [A]
            R = []
            for e in A[2]:
                H = A[0].copy()
                H.remove_edge(e[0], e[1])
                if nx.is_connected(H):
                    shortest_path = nx.single_source_dijkstra(H,0,dest)
                    B = (
                        H,
                        shortest_path[0],
                        nodes_to_edges(shortest_path[1])
                    )
                    for x in helper(B, k-1):
                        R.append(x)
            return R

        while k > 0:
            R = helper(A, k) if k < HTC else helper(A, HTC)
            r = list(map(lambda x: x[1], R))
            if len(r) == 0:
                break
            A = R[r.index(max(r))]
            k -= HTC

        return A
    
    # The main idea here is to come up with a shortest-path-tree 
    # by removing edges/cities from parent graph's shortest path.
    # Use Heuristic, HTC which determines how deep I go down the tree 
    # since going all the way down the tree takes up too much resources.
    # e.g. height of tree = 5, HTC = 2:
    # I go down the tree 2 height then amongst the graphs I have so far, I find the one w/ max shortest path. 
    # Then go 2 more down the tree and so on until I reach the bottom of the tree. 
    # Then compare and go w/ max shortest path.

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
    answer = (G, dijkstra[0])
    A = (G, dijkstra[0], nodes_to_edges(dijkstra[1]))
    if answer[1] <= A[1]:
        A = solver(A, num_k)
        if answer[1] < A[1]:
            answer = A
    k = [e for e in G.edges if e not in answer[0].edges]

    # Remove cities
    for cc in range(num_c):
        less_cities = graph_generator(answer[0])
        for g in less_cities:
            A = (g, nx.dijkstra_path_length(g, 0, dest),
                 nodes_to_edges(nx.dijkstra_path(g, 0, dest)))
            if answer[1] < A[1]:
                answer = A
    c = [v for v in G.nodes if v not in answer[0].nodes]

    return c, k


# Here's an example of how to run your solver.
set_type = 'medium'
input_path = 'inputs/' + set_type + '/'

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    filename = sys.argv[1]
    G = read_input_file(input_path + filename)
    c, k = solve(G)
    assert is_valid_solution(G, c, k)
    print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
    write_output_file(G, c, k, 'singleOutputs/' + set_type + '/' + filename[:-3] + '.out')



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
