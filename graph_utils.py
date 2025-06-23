import jax.numpy as jnp
import jax


def generate_complete_graph(n):
    L = -jnp.ones((n,n))
    L = L.at[jnp.diag_indices(n)].set(n - 1)
    return L

def generate_star_graph(n):
    L = jnp.zeros((n,n))
    #L = np.zeros((n,n))
    L = L.at[0,0].set(n - 1)
    L = L.at[0,1:].set(-1)
    L = L.at[1:,0].set(-1)
    L = L.at[jnp.diag_indices(n)[0][1:],jnp.diag_indices(n)[1][1:]].set(1)

    return L.astype(int)

def generate_ring_graph(n):

    diag = jnp.ones(n) *2
    off_diag = jnp.zeros((n, n))

    def body(i, arr):
        arr = arr.at[i, (i + 1) % n].set(-1)
        arr = arr.at[i, (i - 1) % n].set(-1)
        return arr
    off_diag = jax.lax.fori_loop(0, n, body, off_diag)
    L = off_diag.at[jnp.diag_indices(n)].set(diag)
    return L.astype(int)

def generate_line_graph(n):

    L = jnp.zeros((n,n))
    L[0,0]=1
    L[0,1]=-1
    L[-1,-1]=1
    L[-1,-2]=-1
    
    
    for i in range(1,n-1):
        L[i,i]=2
        L[i,i-1]=-1
        L[i,i+1]=-1
        
        
    return L.astype(int)

def get_graph(graph_type,num_agents):
    if graph_type=="ring":
        return generate_ring_graph(int(num_agents))
    if graph_type=="star":
        return generate_star_graph(int(num_agents))
    if graph_type=="complete":
        return generate_complete_graph(int(num_agents))


def generate_mixing_matrix(graph):
    n = graph.shape[0]
    degrees = jnp.diag(graph)
    
    def compute_row(i):
        row = jnp.zeros(n)
        deg_i = degrees[i]

        def compute_entry(j, row):
            deg_j = degrees[j]
            weight = jnp.where(graph[i,j]==-1, 0.5/ jnp.maximum(deg_i, deg_j),0.0)
            return row.at[j].set(weight)
        row = jax.lax.fori_loop(0, n, compute_entry, row)
        row = row.at[i].set(1.0 - jnp.sum(row))
        return row
    W = jax.vmap(compute_row)(jnp.arange(n))
    return W

if __name__ == "__main__":
    n = 5
    print(generate_complete_graph(n))
    print(generate_star_graph(n))
    print(generate_ring_graph(n))
    print(get_graph("ring",n))
    print(get_graph("star",n))
    # print(get_graph("line",n))
    # print(get_graph("complete",n))

    graph = get_graph("ring", n)
    W = generate_mixing_matrix(graph)
    print(graph)
    print(W)