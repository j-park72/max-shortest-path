import networkx as nx
import glob
import parse
from os.path import basename, normpath

def is_valid_solution(G, c, k):
    """
    Checks whether D is a valid mapping of G, by checking every room adheres to the stress budget.
    Args:
        G: networkx.Graph
        c: List of cities to remove
        k: List of edges to remove (List of tuples)
    Returns:
        bool: false if removing k and c disconnects the graph
    """
    size = len(G)
    H = G.copy()

    for road in k:
        assert H.has_edge(road[0], road[1]), "Invalid Solution: {} is not a valid edge in graph G".format(road)
    H.remove_edges_from(k)
    
    for city in c:
        assert H.has_node(city), "Invalid Solution: {} is not a valid node in graph G".format(city)
    H.remove_nodes_from(c)
    
    assert H.has_node(0), 'Invalid Solution: Source vertex is removed'
    assert H.has_node(size - 1), 'Invalid Solution: Target vertex is removed'

    return nx.is_connected(H)

def calculate_score(G, c, k):
    """
    Calculates the difference between the original shortest path and the new shortest path.
    Args:
        G: networkx.Graph
        c: list of cities to remove
        k: list of edges to remove
    Returns:
        float: total score
    """
    H = G.copy()
    assert is_valid_solution(H, c, k)
    node_count = len(H.nodes)
    original_min_dist = nx.dijkstra_path_length(H, 0, node_count-1)
    H.remove_edges_from(k)
    H.remove_nodes_from(c)
    final_min_dist = nx.dijkstra_path_length(H, 0, node_count-1)
    difference = final_min_dist - original_min_dist
    return difference



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

def compare_scores(set_size, path1, path2):
    difference = {}
    input_path = 'inputs/' + set_size + '/'
    for input_path in glob.glob(input_path + '*'):
        name = basename(normpath(input_path))[:-3]
        output_path = 'tempOutputs/' + set_size + '/' + name + '.out'
        G = parse.read_input_file(input_path)
        g1 = parse.read_output_file(G, path1 + '/' + set_size + '/' + name + '.out')
        g2 = parse.read_output_file(G, path2 + '/' + set_size + '/' + name + '.out')
        difference[name] = (g1-g2,g1,g2)
    return difference
