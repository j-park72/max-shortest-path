import networkx as nx
import sys
from os.path import basename, normpath
import glob



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
            elif (str(A), c) in cache:
                return cache[(str(A), c)]
            R = []
            for i, n in enumerate(A):
                if len(A)-i >= c:
                    for l in get_removeables(A[i+1:], c-1):
                        R.append([n] + l)

            cache[(str(A), c)] = R

            return R

        return helper(A, c)

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
        edges = []

        if len(nodes) == 0:
            return edges

        prev = nodes[0]
        for n in nodes[1:]:
            edges.append((prev, n))
            prev = n

        return edges

    def solver(A, k):

        HTC = 3  # Heuristic - height of a tree

        def helper(A, k):
            if k == 0:
                return [A]
            R = []
            for e in A[2]:
                H = A[0].copy()
                H.remove_edge(e[0], e[1])
                if nx.is_connected(H):
                    B = (
                        H,
                        nx.dijkstra_path_length(H, 0, dest),
                        nodes_to_edges(nx.dijkstra_path(H, 0, dest))
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

    # while k > 0, get shortest path s-t path, for e in s-t path edges remove e then get shortest path.
    # Then compare and go w/ max shortest path.

    # Initialize
    num_k, num_c, dest = 0, 0, G.number_of_nodes()-1
    if G.number_of_nodes() <= 30:
        num_k, num_c = 15, 1
    elif G.number_of_nodes() <= 50:
        num_k, num_c = 50, 3
    elif G.number_of_nodes() <= 100:
        num_k, num_c = 100, 5

    answer = (G, nx.dijkstra_path_length(G, 0, dest))
    A = (G, nx.dijkstra_path_length(G, 0, dest),
         nodes_to_edges(nx.dijkstra_path(G, 0, dest)))
    if answer[1] <= A[1]:
        A = solver(A, num_k)
        if answer[1] < A[1]:
            answer = A
    k = [e for e in G.edges if e not in answer[0].edges]

    for cc in range(num_c):
        less_cities = graph_generator(answer[0])
        for g in less_cities:
            A = (g, nx.dijkstra_path_length(g, 0, dest),
                 nodes_to_edges(nx.dijkstra_path(g, 0, dest)))
            if answer[1] < A[1]:
                answer = A
    c = [v for v in G.nodes if v not in answer[0].nodes]

    return c, k
