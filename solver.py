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
            list of nodes/edges to remove
        """

        cache = {}

        def helper(A, c):
            if len(A) == 0 or c == 0 or len(A) < c:
                return [[]]
            elif (str(A),c) in cache:
                return cache[(str(A),c)]
            R = []
            for i,n in enumerate(A):
                if len(A)-i >= c:
                    for l in get_removeables(A[i+1:], c-1):
                        R.append([n] + l)

            cache[(str(A),c)] = R

            return R

        return helper(A,c)
    
    def graph_generator(H, k):
        """
        Args:
            H: networkx.Graph
            k: # of cities to remove
        Returns:
            L: list of all possible connected graphs w/ (H.nodes - k) cities 
        """
        L = []
        nodes = list(H.nodes)
        nodes.remove(0)
        nodes.remove(dest)
        for nodes_to_remove in get_removeables(nodes, k):
            h = H.copy()
            h.remove_nodes_from(nodes_to_remove)
            if nx.is_connected(h):
                L.append(h) 

        return L 
    
    def nodes_to_edges(nodes):
        edges = []

        if len(nodes) == 0:
            return edges
        
        prev = nodes[0]
        for n in nodes[1:]:
            edges.append((prev,n))
            prev = n

        return edges

    def solver(A, k):

        HTC = 3 # Heuristic

        def helper(A, k):
            if k == 0:
                return [A]
            R = []
            for e in A[2]:
                H = A[0].copy()
                H.remove_edge(e[0],e[1])
                if nx.is_connected(H):
                    B = (
                        H, 
                        nx.dijkstra_path_length(H,0,dest), 
                        nodes_to_edges(nx.dijkstra_path(H,0,dest))
                        )
                        
                    for x in helper(B, k-1):
                        R.append(x)

            return R

        while k > 0:
            R = helper(A,k) if k < HTC else helper(A,HTC)
            r = list(map(lambda x:x[1], R))
            if len(r) == 0:
                break
            A = R[r.index(max(r))]
            k -= HTC

        return A

    # while k > 0, get shortest path s-t path, for e in s-t path edges remove e then get shortest path.
    # Then compare and go w/ max shortest path.

    # Initialize
    num_k,num_c,dest = 0,0,G.number_of_nodes()-1
    if G.number_of_nodes() <= 30:
        num_k,num_c = 15,1
    elif G.number_of_nodes() <= 50:
        num_k,num_c = 50,3
    elif G.number_of_nodes() <= 100:
        num_k,num_c = 100,5

    answer = (G, nx.dijkstra_path_length(G,0,dest))
    for cc in range(num_c+1):
        less_cities = graph_generator(G, cc)
        for g in less_cities:
            A = (g, nx.dijkstra_path_length(g,0,dest), 
                nodes_to_edges(nx.dijkstra_path(g,0,dest)))
            if answer[1] <= A[1]:
                A = solver(A, num_k)
                if answer[1] < A[1]:
                    answer = A

    c = [v for v in G.nodes if v not in answer[0].nodes]
    k = [e for e in G.edges if e not in answer[0].edges]
    return c,k
    




def naive_solve(G):
    """
    This function will terminate on G with many edges. "many" as in < 1000.
    Correct algorithm but since the problem is NP-Hard, 
    but not very useful algorithm cause it takes forever.

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
            list of nodes/edges to remove
        """

        cache = {}

        def helper(A, c):
            if len(A) == 0 or c == 0 or len(A) < c:
                return [[]]
            elif (str(A),c) in cache:
                return cache[(str(A),c)]
            R = []
            for i,n in enumerate(A):
                if len(A)-i >= c:
                    for l in get_removeables(A[i+1:], c-1):
                        R.append([n] + l)

            cache[(str(A),c)] = R

            return R

        return helper(A,c)


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
    for cc in range(num_c+1):
        less_cities = graph_generator(G, cc, True)
        for g in less_cities:
            # Run Dijkstra's Algorithm on every possible route on g w/ up to num_k removed edges
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
distance = {}

input_path = 'inputs/inputs/small/*' 
if __name__ == '__main__':
    inputs = glob.glob(input_path)
    for input_path in inputs:
        output_path = 'outputs/small/' + basename(normpath(input_path))[:-3] + '.out'
        G = read_input_file(input_path)
        c, k = solve(G)
        assert is_valid_solution(G, c, k)
        distance[input_path] = calculate_score(G, c, k)
        write_output_file(G, c, k, output_path)


