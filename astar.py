import heapq
import time

def manhattan(p, goal):
    """
    Heuristique de Manhattan pour une grille en 4-connexité.
    h((x,y)) = |x - x_g| + |y - y_g|
    Admissible si chaque déplacement coûte 1 (ne surestime jamais).
    Cohérente : h(n) <= c(n,n') + h(n')  pour tout successeur n'.
    """
    return abs(p[0] - goal[0]) + abs(p[1] - goal[1])


def zero_heuristic(p, goal):
    """
    Heuristique nulle h=0.
    Dégénère A* en UCS (Dijkstra) — admissible triviale mais non informée.
    """
    return 0


def neighbors(state, grid):
    """
    Retourne la liste des états voisins valides (4-connexité) depuis state.

    Paramètres
    ----------
    state : tuple (row, col)
    grid  : liste 2D ou np.ndarray, 0 = libre, 1 = obstacle

    Retourne
    --------
    voisins : liste de tuples (row, col) accessibles
    """
    rows = len(grid)
    cols = len(grid[0])
    r, c = state
    result = []
    # 4-voisins : haut, bas, gauche, droite
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            result.append((nr, nc))
    return result


# -----------------------------------------------------------------------------
# OPEN = heap min sur f(n),  CLOSED = set des états développés
# -----------------------------------------------------------------------------

def astar(grid, start, goal, h=manhattan, weight=1.0):
    """
    Algorithme A* générique sur grille 2D à coût uniforme.

    f(n) = g(n) + weight * h(n)
      weight = 1.0  → A* standard (optimal si h admissible)
      weight = 0.0  → UCS / Dijkstra  (f = g uniquement)
      weight = inf  → Greedy Best-First (f = h uniquement)

    Paramètres
    ----------
    grid   : grille 2D (0=libre, 1=obstacle)
    start  : état initial (row, col)
    goal   : état but    (row, col)
    h      : heuristique callable(state, goal) → float
    weight : coefficient w pour Weighted A*

    Retourne
    --------
    path     : liste ordonnée d'états de start à goal ([] si échec)
    g_cost   : coût total du chemin (float)
    expanded : nombre de nœuds développés (sortis de OPEN)
    open_max : taille maximale de OPEN observée
    elapsed  : temps CPU en secondes
    stats    : dictionnaire récapitulatif
    """
    t0 = time.perf_counter()

    # OPEN : tas min sur (f, g, état)
    # On ajoute un tie-breaker sur h pour départager les nœuds de même f
    open_heap = []
    h_start = h(start, goal)
    heapq.heappush(open_heap, (h_start, 0.0, start))

    # g(n) : meilleur coût connu pour atteindre chaque état
    g_score = {start: 0.0}

    # Reconstruction du chemin : parent de chaque état
    came_from = {start: None}

    # CLOSED : états déjà développés (on ne les ré-explore pas)
    closed = set()

    expanded = 0
    open_max = 1

    while open_heap:
        open_max = max(open_max, len(open_heap))
        f_cur, g_cur, current = heapq.heappop(open_heap)

        # Si déjà développé avec un meilleur coût, on ignore
        if current in closed:
            continue

        closed.add(current)
        expanded += 1

        # ── Test du but ──────────────────────────────────────────────────
        if current == goal:
            path = _reconstruct_path(came_from, goal)
            elapsed = time.perf_counter() - t0
            stats = _make_stats("A*", g_cur, path, expanded, open_max, elapsed)
            return path, g_cur, expanded, open_max, elapsed, stats

        # ── Développement des voisins ────────────────────────────────────
        for neighbor in neighbors(current, grid):
            new_g = g_cur + 1          # coût uniforme c(n, n') = 1

            if neighbor in closed:
                continue

            if new_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_g
                came_from[neighbor] = current
                h_val = h(neighbor, goal)
                # Greedy pur : f = h seulement (weight → ∞ simulé par mode)
                f_val = new_g + weight * h_val
                heapq.heappush(open_heap, (f_val, new_g, neighbor))

    # Aucun chemin trouvé
    elapsed = time.perf_counter() - t0
    stats = _make_stats("A*", float('inf'), [], expanded, open_max, elapsed)
    return [], float('inf'), expanded, open_max, elapsed, stats


def ucs(grid, start, goal):
    """
    Uniform Cost Search (Dijkstra) : f(n) = g(n)   [h = 0]
    Optimal mais ne guide pas la recherche vers le but.
    """
    path, cost, expanded, open_max, elapsed, stats = astar(
        grid, start, goal, h=zero_heuristic, weight=1.0)
    stats["algorithm"] = "UCS"
    return path, cost, expanded, open_max, elapsed, stats


def greedy(grid, start, goal):
    """
    Greedy Best-First : f(n) = h(n)   [g ignoré]
    Rapide mais non optimal — peut trouver un chemin sous-optimal.
    Implémenté en fixant g_score à 0 dans f (on passe weight très grand).
    """
    # Astuce : on redéfinit l'heuristique comme f complet (g non utilisé)
    def h_only(p, g):
        return manhattan(p, g)

    # Pour ignorer g, on utilise weight=1 mais on réécrit f = h dans une
    # version dédiée qui n'additionne pas g.
    t0 = time.perf_counter()
    open_heap = []
    h_start = manhattan(start, goal)
    heapq.heappush(open_heap, (h_start, start))
    came_from = {start: None}
    visited   = set()
    expanded  = 0
    open_max  = 1

    while open_heap:
        open_max = max(open_max, len(open_heap))
        h_cur, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        expanded += 1

        if current == goal:
            path = _reconstruct_path(came_from, goal)
            g_cost  = len(path) - 1           # coût réel (même si non minimisé)
            elapsed = time.perf_counter() - t0
            stats   = _make_stats("Greedy", g_cost, path, expanded, open_max, elapsed)
            return path, g_cost, expanded, open_max, elapsed, stats

        for nb in neighbors(current, grid):
            if nb not in visited and nb not in came_from:
                came_from[nb] = current
                heapq.heappush(open_heap, (manhattan(nb, goal), nb))

    elapsed = time.perf_counter() - t0
    stats   = _make_stats("Greedy", float('inf'), [], expanded, open_max, elapsed)
    return [], float('inf'), expanded, open_max, elapsed, stats


def astar_manhattan(grid, start, goal):
    """
    A* standard avec heuristique Manhattan (weight=1).
    Optimal et complet si h est admissible (ce qui est le cas ici).
    """
    path, cost, expanded, open_max, elapsed, stats = astar(
        grid, start, goal, h=manhattan, weight=1.0)
    stats["algorithm"] = "A*"
    return path, cost, expanded, open_max, elapsed, stats


def weighted_astar(grid, start, goal, w=2.0):
    """
    Weighted A* : f(n) = g(n) + w * h(n),  w > 1.
    Compromis : moins de nœuds développés, mais chemin ε-sous-optimal.
    Garantie : coût_trouvé <= w * coût_optimal  (si h admissible).
    """
    path, cost, expanded, open_max, elapsed, stats = astar(
        grid, start, goal, h=manhattan, weight=w)
    stats["algorithm"] = f"Weighted A* (w={w})"
    return path, cost, expanded, open_max, elapsed, stats


def extract_policy(path):
    """
    Construit la politique π à partir du chemin planifié par A*.

    À chaque état du chemin, associe l'action (direction) qui mène
    à l'état suivant.  Le dernier état (but) reçoit l'action 'STAY'.

    Paramètres
    ----------
    path : liste ordonnée d'états [(r0,c0), (r1,c1), ..., (rn,cn)]

    Retourne
    --------
    policy : dict  état -> (action_str, état_suivant)
        action_str ∈ {'UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'}
    """
    if not path:
        return {}

    DELTA_TO_ACTION = {
        (-1,  0): 'UP',
        ( 1,  0): 'DOWN',
        ( 0, -1): 'LEFT',
        ( 0,  1): 'RIGHT',
    }

    policy = {}
    for i in range(len(path) - 1):
        s      = path[i]
        s_next = path[i + 1]
        dr = s_next[0] - s[0]
        dc = s_next[1] - s[1]
        action = DELTA_TO_ACTION[(dr, dc)]
        policy[s] = (action, s_next)

    # État but : action STAY (absorbant dans la chaîne de Markov)
    policy[path[-1]] = ('STAY', path[-1])
    return policy


def _reconstruct_path(came_from, goal):
    """Remonte le dictionnaire came_from pour reconstruire le chemin."""
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def _make_stats(algorithm, cost, path, expanded, open_max, elapsed):
    return {
        "algorithm"  : algorithm,
        "path_cost"  : cost,
        "path_length": len(path),
        "expanded"   : expanded,
        "open_max"   : open_max,
        "elapsed_ms" : elapsed * 1000,
    }