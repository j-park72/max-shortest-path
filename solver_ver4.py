import networkx as nx
import sys
from utils import calculate_score
from os.path import basename, normpath
import glob


def solve(G):
    """
    The main idea here is to come up with a shortest-path-tree
    by removing edges/cities from parent graph's shortest path.
    Use heuristic variable, HTC which determines how deep I go down the tree
    since going all the way down the tree takes up too much resources.
    e.g. height of tree = 5, HTC = 2:
    I go down the tree 2 height then amongst the graphs I have so far, I find the one w/ max shortest path.
    Then go 2 more down the tree and so on until I reach the bottom of the tree.
    Then compare and go w/ max shortest path.

    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """

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
                    shortest_path = nx.single_source_dijkstra(H, 0, dest)
                    B = (H, shortest_path[0], nodes_to_edges(shortest_path[1]))
                    for x in down_edge_tree(B, k-1):
                        R.append(x)
            if len(R) > 0:
                r = list(map(lambda x: x[1], R))
                R = [R[r.index(max(r))]]
            return R

        def down_node_tree(A, k):
            if k == 0:
                return [A]
            R = []
            for n in A[2][1:-1]:
                H = A[0].copy()
                H.remove_node(n)
                if nx.is_connected(H):
                    shortest_path = nx.single_source_dijkstra(H, 0, dest)
                    B = (H, shortest_path[0], shortest_path[1])
                    for x in down_node_tree(B, k-1):
                        R.append(x)
            if len(R) > 0:
                r = list(map(lambda x: x[1], R))
                R = [R[r.index(max(r))]]
            return R

        recurser = down_edge_tree if is_edges else down_node_tree

        R = recurser(A, k)

        return R[0] if len(R) > 0 else A

    # Initialize
    num_k, num_c, dest = 0, 0, G.number_of_nodes()-1
    if G.number_of_nodes() <= 30:
        num_k, num_c = 15, 1
    elif G.number_of_nodes() <= 50:
        num_k, num_c = 50, 3
    elif G.number_of_nodes() <= 100:
        num_k, num_c = 100, 5
    dijkstra = nx.single_source_dijkstra(G, 0, dest)

    def edges_to_cities():
        # Remove edges
        answer = (G, dijkstra[0], nodes_to_edges(dijkstra[1]))
        A = solver(answer, num_k, True)
        if answer[1] < A[1]:
            answer = A
        k = [e for e in G.edges if e not in answer[0].edges]

        # Remove cities
        answer = (answer[0], answer[1], nx.dijkstra_path(answer[0], 0, dest))
        A = solver(answer, num_c, False)
        if answer[1] < A[1]:
            answer = A
        c = [v for v in G.nodes if v not in answer[0].nodes]
        return c, k

    def cities_to_edges():
        # Remove cities
        answer = (G, dijkstra[0], dijkstra[1])
        A = solver(answer, num_c, False)
        if answer[1] < A[1]:
            answer = A
        c = [v for v in G.nodes if v not in answer[0].nodes]

        # Remove edges
        answer = (answer[0], answer[1], nodes_to_edges(answer[2]))
        A = solver(answer, num_c, True)
        if answer[1] < A[1]:
            answer = A
        k = [e for e in G.edges if e not in answer[0].edges]

        return c, k

    e_to_c = edges_to_cities()
    c_to_e = cities_to_edges()

    return e_to_c if calculate_score(G, e_to_c[0], e_to_c[1]) > calculate_score(G, c_to_e[0], c_to_e[1]) else c_to_e
