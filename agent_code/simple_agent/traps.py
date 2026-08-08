from collections import deque

from .constants import DIRECTIONS
from .blast import get_blast_coords
from .pathfinding import find_reachable_tiles


def find_articulation_points(adj):
    """Return the set of nodes whose removal would disconnect the graph `adj` (iterative Tarjan's algorithm)."""
    disc, low, parent = {}, {}, {}
    ap = set()
    timer = [0]

    def dfs(root):
        stack = [(root, iter(adj[root]))]
        disc[root] = low[root] = timer[0]
        timer[0] += 1
        parent[root] = None
        children_of_root = 0

        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if v not in disc:
                    disc[v] = low[v] = timer[0]
                    timer[0] += 1
                    parent[v] = u
                    if u == root:
                        children_of_root += 1
                    stack.append((v, iter(adj[v])))
                    advanced = True
                    break
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
            if not advanced:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[u])
                    if p != root and low[u] >= disc[p]:
                        ap.add(p)

        if children_of_root > 1:
            ap.add(root)

    for node in adj:
        if node not in disc:
            dfs(node)

    return ap


def connected_components(adj, nodes):
    """Return the connected components of `nodes` within graph `adj`, as a list of sets."""
    nodes = set(nodes)
    seen = set()
    components = []
    for start in nodes:
        if start in seen:
            continue
        comp = set()
        queue = deque([start])
        seen.add(start)
        while queue:
            u = queue.popleft()
            comp.add(u)
            for v in adj[u]:
                if v in nodes and v not in seen:
                    seen.add(v)
                    queue.append(v)
        components.append(comp)
    return components


def find_traps(field, power):
    """Identify dead-end pockets on `field` reachable only through a single chokepoint.

    Returns {tile: (chokepoint, pocket, dist_to_choke, lethal)} for every tile in such a pocket,
    where `lethal` marks whether a bomb of the given `power` at the chokepoint would hit the whole pocket.
    """
    free = {(x, y) for x in range(field.shape[0]) for y in range(field.shape[1])
            if field[x, y] == 0}

    adj = {p: [] for p in free}
    for (x, y) in free:
        for dx, dy in DIRECTIONS:
            n = (x + dx, y + dy)
            if n in free:
                adj[(x, y)].append(n)

    traps = {}
    for c in find_articulation_points(adj):
        components = connected_components(adj, free - {c})
        if len(components) < 2:
            continue
        blast = get_blast_coords(field, c, power)
        components.sort(key=len, reverse=True)
        for pocket in components[1:]:
            pocket = frozenset(pocket)
            lethal = pocket <= blast
            dist_to_choke = find_reachable_tiles(field, c, avoid=free - pocket - {c})
            for t in pocket:
                prev = traps.get(t)
                if prev is None or len(pocket) < len(prev[1]) or (lethal and not prev[3]):
                    traps[t] = (c, pocket, dist_to_choke, lethal)
    return traps
