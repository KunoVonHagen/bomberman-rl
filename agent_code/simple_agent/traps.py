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


def _dfs_analyze(adj):
    disc, low, parent = {}, {}, {}
    tree_children = {}
    enter_order = []
    enter_index = {}
    subtree_size = {}
    root_of = {}
    ap = set()
    timer = [0]

    for root in adj:
        if root in disc:
            continue

        disc[root] = low[root] = timer[0]
        enter_index[root] = len(enter_order)
        enter_order.append(root)
        timer[0] += 1
        parent[root] = None
        tree_children[root] = []
        root_of[root] = root
        children_of_root = 0

        stack = [(root, iter(adj[root]))]
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if v not in disc:
                    disc[v] = low[v] = timer[0]
                    enter_index[v] = len(enter_order)
                    enter_order.append(v)
                    timer[0] += 1
                    parent[v] = u
                    tree_children[v] = []
                    tree_children[u].append(v)
                    root_of[v] = root
                    if u == root:
                        children_of_root += 1
                    stack.append((v, iter(adj[v])))
                    advanced = True
                    break
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
            if not advanced:
                stack.pop()
                subtree_size[u] = 1 + sum(subtree_size[c] for c in tree_children[u])
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[u])
                    if p != root and low[u] >= disc[p]:
                        ap.add(p)

        if children_of_root > 1:
            ap.add(root)

    return {
        "disc": disc,
        "low": low,
        "parent": parent,
        "tree_children": tree_children,
        "enter_order": enter_order,
        "enter_index": enter_index,
        "subtree_size": subtree_size,
        "root_of": root_of,
        "articulation_points": ap,
    }


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

    info = _dfs_analyze(adj)
    disc = info["disc"]
    low = info["low"]
    parent = info["parent"]
    tree_children = info["tree_children"]
    enter_order = info["enter_order"]
    enter_index = info["enter_index"]
    subtree_size = info["subtree_size"]
    root_of = info["root_of"]

    def subtree_nodes(v):
        i = enter_index[v]
        return frozenset(enter_order[i:i + subtree_size[v]])

    traps = {}
    for c in info["articulation_points"]:
        is_root = parent[c] is None

        cutoff_children = [v for v in tree_children[c] if is_root or low[v] >= disc[c]]
        cutoff_sizes = [subtree_size[v] for v in cutoff_children]

        total_size = subtree_size[root_of[c]]
        rest_size = total_size - 1 - sum(cutoff_sizes)

        entries = [(subtree_size[v], v) for v in cutoff_children]
        if rest_size > 0:
            entries.append((rest_size, None))

        if len(entries) < 2:
            continue

        entries.sort(key=lambda e: e[0], reverse=True)
        pocket_entries = entries[1:]

        blast = get_blast_coords(field, c, power)
        rest_set = None

        for _, marker in pocket_entries:
            if marker is not None:
                pocket = subtree_nodes(marker)
            else:
                if rest_set is None:
                    root_nodes = subtree_nodes(root_of[c])
                    cutoff_union = set()
                    for v in cutoff_children:
                        cutoff_union.update(subtree_nodes(v))
                    rest_set = frozenset((root_nodes - cutoff_union) - {c})
                pocket = rest_set

            lethal = pocket <= blast
            dist_to_choke = find_reachable_tiles(field, c, avoid=free - pocket - {c})
            for t in pocket:
                prev = traps.get(t)
                if prev is None or len(pocket) < len(prev[1]) or (lethal and not prev[3]):
                    traps[t] = (c, pocket, dist_to_choke, lethal)

    return traps