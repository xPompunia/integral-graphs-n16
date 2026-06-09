import sys
import networkx as nx
import numpy as np

def todokuwiki(G, graf6, Sp):
    n = nx.number_of_nodes(G)
    s = '<graphviz circo>\ngraph g{'
    for i in range(n):
        s += str(i+1) + ';'
    for e in G.edges:
        s += str(e[0]+1) + '--' + str(e[1]+1) + ';'
    s += 'label="k:' + str(len(G.edges)) + ', g6:\\"' + graf6 + '\\",\n Sp=' + Sp + '";\n'
    s += '}\n</graphviz>'
    return s

if __name__ == "__main__":
    graf6 = sys.argv[1] if (len(sys.argv) > 1) else "@"
    G = nx.from_graph6_bytes(graf6.strip().encode("utf-8"))
    Sp = "[" + ",".join(["{:.4f}".format(float(i)) for i in reversed(np.linalg.eigvalsh(nx.to_numpy_array(G)))]) + "]"
    print(todokuwiki(G, graf6, Sp))
