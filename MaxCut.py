import cvxpy as cvx
import networkx as nx
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def load_graph_from_file(file_path: str) -> nx.Graph:
   G = nx.Graph()
   with open(file_path, 'r') as file:
       for line in file:
           # Decrement node identifiers to convert to zero-based indexing
           node1, node2, *_ = map(lambda x: int(x) - 1, line.strip().split())
           G.add_edge(node1, node2)
   return G

def parallel_random_vector_rounding(sdp_vectors, seed):
   np.random.seed(seed)
   random_vector = np.random.randn(sdp_vectors.shape[1])
   random_vector /= np.linalg.norm(random_vector)
   colors = np.sign(np.dot(sdp_vectors, random_vector))
   return colors

def goemans_williamson_with_enhancements(graph: nx.Graph, rounding_attempts=2000000):
   laplacian = 0.25 * nx.laplacian_matrix(graph).todense()
   psd_mat = cvx.Variable(laplacian.shape, PSD=True)
   obj = cvx.Maximize(cvx.trace(laplacian @ psd_mat))
   constraints = [cvx.diag(psd_mat) == 1]
   prob = cvx.Problem(obj, constraints)
   prob.solve(solver=cvx.SCS, verbose=False, max_iters=1000000, eps=1e-8)

   evals, evects = np.linalg.eigh(psd_mat.value)
   sdp_vectors = evects.T[evals > 1e-8].T

   # Parallel rounding process
   with ProcessPoolExecutor() as executor:
      results = executor.map(parallel_random_vector_rounding, 
      [sdp_vectors]*rounding_attempts, 
      range(rounding_attempts))
   scores_and_colors = [(np.dot(colors.T, np.dot(laplacian, colors)), colors) for colors in results]
   best_score, best_colors = max(scores_and_colors, key=lambda x: x[0])

   best_colors = local_search_improvement(graph, best_colors)
   return best_colors, best_score

def local_search_improvement(graph, initial_colors):
   nodes_by_degree = sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True)
   node_index_map = {node: i for i, node in enumerate(nodes_by_degree)}
   best_colors = initial_colors.copy()
   best_score = calculate_cut_value(graph, best_colors, node_index_map)

   improved = True
   while improved:
      improved = False
      for node in nodes_by_degree:
         node_idx = node_index_map[node]
         current_colors = best_colors.copy()
         current_colors[node_idx] = -current_colors[node_idx]
         current_score = calculate_cut_value(graph, current_colors, node_index_map)
         if current_score > best_score:
            best_score = current_score
            best_colors = current_colors.copy()
            improved = True
   return best_colors

def calculate_cut_value(graph, colors, node_index_map):
   score = 0
   for edge in graph.edges():
      if colors[node_index_map[edge[0]]] != colors[node_index_map[edge[1]]]:
         score += 1
   return score

def save_results_to_file(file_path, score, binary_solution):
    with open(file_path, 'a') as file:
        file.write(f"{score} {binary_solution}\n")

def save_binary_solution_to_file(file_path, binary_solution):
    with open(file_path, 'a') as file:
        file.write(f"{binary_solution}\n")

   '''
   OUTPUT SOLUTIONS
   '''
if __name__ == '__main__':
    results_file_path = 'results_400.txt'
    binary_solutions_file_path = 'binary_solutions_400.txt'
    
    for i in range(30):
         file_path = f'./data/syn/powerlaw_400_ID{i}.txt'
         print(file_path)
         G = load_graph_from_file(file_path)
         colors, score = goemans_williamson_with_enhancements(G)
         binary_solution = ''.join('1' if color == 1 else '0' for color in colors)[:400]
         
         # Save results to results file
         save_results_to_file(results_file_path, score, binary_solution)
         
         # Save binary solution to binary solutions file
         save_binary_solution_to_file(binary_solutions_file_path, binary_solution)
