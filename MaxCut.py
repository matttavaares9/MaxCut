import cvxpy as cvx
import networkx as nx
import numpy as np

def load_graph_from_file(file_path: str) -> nx.Graph:
    G = nx.Graph()
    with open(file_path, 'r') as file:
        for line in file:
            node1, node2, *_ = map(int, line.strip().split())
            G.add_edge(node1, node2)
    return G

def goemans_williamson_with_enhancements(graph: nx.Graph, rounding_attempts=10000) -> (np.ndarray, float):
    best_score = -np.inf
    best_colors = None
    best_solution = None

    for _ in range(rounding_attempts):
        laplacian = 0.25 * nx.laplacian_matrix(graph).toarray()
        psd_mat = cvx.Variable(laplacian.shape, PSD=True)
        obj = cvx.Maximize(cvx.trace(laplacian @ psd_mat))
        constraints = [cvx.diag(psd_mat) == 1]
        prob = cvx.Problem(obj, constraints)
        
        # Adjust solver options for better performance with large datasets
        prob.solve(solver=cvx.SCS, verbose=True, max_iters=10000, eps=1e-6)

        evals, evects = np.linalg.eigh(psd_mat.value)
        sdp_vectors = evects[:, evals > 1e-6]

        random_vector = np.random.randn(sdp_vectors.shape[1])
        random_vector /= np.linalg.norm(random_vector)
        colors = np.sign(sdp_vectors @ random_vector)
        score = np.dot(colors.T, np.dot(laplacian, colors))

        if score > best_score:
            best_score = score
            best_colors = colors
            best_solution = best_colors.copy()

    best_solution = local_search_improvement(graph, best_solution)
    return best_solution, best_score

def local_search_improvement(graph: nx.Graph, initial_colors: np.ndarray) -> np.ndarray:
    nodes = list(graph.nodes())
    node_index_map = {node: i for i, node in enumerate(nodes)}
    best_colors = initial_colors.copy()
    best_score = calculate_cut_value(graph, best_colors, node_index_map)

    improved = True
    while improved:
        improved = False
        random.shuffle(nodes)
        for node in nodes:
            node_idx = node_index_map[node]
            current_colors = best_colors.copy()
            current_colors[node_idx] = -current_colors[node_idx]  
            current_score = calculate_cut_value(graph, current_colors, node_index_map)
            if current_score > best_score:
                best_score = current_score
                best_colors = current_colors.copy()
                improved = True

    return best_colors

def calculate_cut_value(graph: nx.Graph, colors: np.ndarray, node_index_map: dict) -> float:
    score = 0
    for edge in graph.edges():
        if colors[node_index_map[edge[0]]] != colors[node_index_map[edge[1]]]:
            score += 1
    return score

if __name__ == '__main__':
    for(i in range(0,29))
    file_path = './data/syn/powerlaw_{NODES}_ID{ID}.txt'.format("100","0")
    G = load_graph_from_file(file_path)

    best_solution, best_score = goemans_williamson_with_enhancements(G, rounding_attempts=10000)

    binary_solution = ''.join(['1' if color == 1 else '0' for color in best_solution])
    binary_solution = binary_solution[:800]

    print(f"Cut-value: {best_score}")
    print(f"Binary Solution Vector: {binary_solution}")