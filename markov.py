import numpy as np
from collections import defaultdict, deque

# ── États absorbants (noms symboliques) ──────────────────────────────────────
GOAL = "GOAL"
FAIL = "FAIL"

# ── Deltas directionnels ─────────────────────────────────────────────────────
ACTION_DELTA = {
    'UP'   : (-1,  0),
    'DOWN' : ( 1,  0),
    'LEFT' : ( 0, -1),
    'RIGHT': ( 0,  1),
    'STAY' : ( 0,  0),
}

# Directions latérales perpendiculaires à chaque action
LATERAL = {
    'UP'   : ['LEFT', 'RIGHT'],
    'DOWN' : ['LEFT', 'RIGHT'],
    'LEFT' : ['UP',   'DOWN' ],
    'RIGHT': ['UP',   'DOWN' ],
    'STAY' : [],
}


# =============================================================================
# — Construction de la chaîne de Markov 
# =============================================================================

def build_transition_matrix(grid, policy, epsilon=0.1):
    """
    — Construire la matrice de transition P à partir de la politique et
            du niveau d'incertitude ε 

    Modèle stochastique (paramétré par ε) :
      • action voulue vers n'   : probabilité 1 − ε
      • déviation latérale ×2  : ε/2 chacune
      • si collision → reste sur place (ou état FAIL absorbant)

    États de la matrice :
      tous les états du chemin  +  GOAL  +  FAIL

    Paramètres
    ----------
    grid    : grille 2D numpy (0=libre, 1=obstacle)
    policy  : dict état → (action, état_suivant)   (sortie de extract_policy)
    epsilon : taux de déviation ε ∈ [0, 1]

    Retourne
    --------
    P         : np.ndarray (n_états × n_états)  — matrice stochastique
    states    : liste ordonnée des états (dont GOAL et FAIL en fin)
    state_idx : dict état → indice entier
    """
    rows, cols = len(grid), len(grid[0])

    # Ordre des états : états du chemin (policy keys) + GOAL + FAIL
    path_states = [s for s in policy if s not in (GOAL, FAIL)]
    all_states  = path_states + [GOAL, FAIL]
    state_idx   = {s: i for i, s in enumerate(all_states)}
    n           = len(all_states)

    P = np.zeros((n, n))

    # ── Remplissage de P ──────────────────────────────────────────────────────
    for s, (action, _) in policy.items():
        i = state_idx[s]

        # ─ État but : transition vers GOAL (absorbant) ─
        if action == 'STAY':
            P[i, state_idx[GOAL]] = 1.0
            continue

        # ─ Calcul des cellules cibles ─
        def target(act):
            """Cellule résultante si on applique `act` depuis `s`."""
            dr, dc = ACTION_DELTA[act]
            nr, nc = s[0] + dr, s[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                return (nr, nc)   # déplacement réussi
            return None           # collision / hors-grille

        main_cell = target(action)
        lat_cells = [target(d) for d in LATERAL[action]]

        # ─ Fonction d'attribution de probabilité ─
        def add_prob(cell, prob):
            if cell is None:
                # Collision → état FAIL (section 3.3)
                P[i, state_idx[FAIL]] += prob
            elif cell in state_idx:
                P[i, state_idx[cell]] += prob
            else:
                # Cellule hors politique → FAIL par défaut
                P[i, state_idx[FAIL]] += prob

        # Attribution selon le modèle à 3 issues
        add_prob(main_cell, 1.0 - epsilon)
        if len(lat_cells) == 2:
            add_prob(lat_cells[0], epsilon / 2.0)
            add_prob(lat_cells[1], epsilon / 2.0)

    # ── États absorbants (section 3.3) ───────────────────────────────────────
    for absorb in [GOAL, FAIL]:
        idx = state_idx[absorb]
        P[idx, :] = 0.0
        P[idx, idx] = 1.0

    # ── P3.2 — Vérification stochasticité ────────────────────────────────────
    _verify_stochastic(P, all_states)

    return P, all_states, state_idx


def _verify_stochastic(P, states):
    """
    — Vérifie que P est stochastique (somme de chaque ligne = 1).
    Corrige par normalisation en cas d'écart numérique.
    """
    row_sums = P.sum(axis=1)
    for i in range(len(states)):
        s = row_sums[i]
        if abs(s - 1.0) > 1e-9:
            if s > 0:
                P[i] /= s          # normalisation
            else:
                P[i, i] = 1.0      # état piège absorbant sur lui-même


# =============================================================================
# — Évolution de la distribution : π^(n) = π^(0) · P^n
# =============================================================================

def compute_pi_n(pi0, P, n_steps):
    """
    — Calcule π^(n) = π^(0) · P^n pour n = 0, 1, …, n_steps
            par multiplications successives (Chapman-Kolmogorov).

    Paramètres
    ----------
    pi0     : np.ndarray (n_états,) — distribution initiale
    P       : np.ndarray (n_états × n_états)
    n_steps : nombre d'étapes à calculer

    Retourne
    --------
    history : np.ndarray (n_steps+1, n_états)
              history[t] = π^(t)
    """
    n_states = len(pi0)
    history  = np.zeros((n_steps + 1, n_states))
    history[0] = pi0.copy()
    pi = pi0.copy()
    for t in range(1, n_steps + 1):
        pi = pi @ P          # π^(t) = π^(t-1) · P
        history[t] = pi
    return history


# =============================================================================
# — Analyse Markov : graphe, classes, absorption, période
# =============================================================================

def build_transition_graph(P, states, threshold=1e-9):
    """
    — Construit le graphe orienté des transitions.
    Arc i → j existe ssi P[i,j] > threshold.

    Retourne
    --------
    graph : dict état → liste d'états successeurs
    """
    graph = defaultdict(list)
    n = len(states)
    for i in range(n):
        for j in range(n):
            if P[i, j] > threshold:
                graph[states[i]].append(states[j])
    return dict(graph)


def find_communication_classes(P, states):
    """
    — Identifie les classes de communication par SCC (Kosaraju).

    Deux états i et j communiquent (i ↔ j) ssi i → j et j → i.
    Une classe est récurrente (persistante) si elle est fermée
    (aucun arc n'en sort), transitoire sinon.

    Retourne
    --------
    classes    : liste de sets, chaque set est une SCC
    class_type : dict état → 'Récurrent' | 'Transitoire'
    class_of   : dict état → indice de sa classe
    """
    n = len(states)
    idx_of = {s: i for i, s in enumerate(states)}

    # Passe 1 : DFS forward, ordre de fin
    visited = [False] * n
    finish_order = []

    def dfs1(v):
        stack = [(v, iter(range(n)))]
        visited[v] = True
        while stack:
            node, children = stack[-1]
            try:
                w = next(children)
                if P[node, w] > 1e-9 and not visited[w]:
                    visited[w] = True
                    stack.append((w, iter(range(n))))
            except StopIteration:
                finish_order.append(node)
                stack.pop()

    for v in range(n):
        if not visited[v]:
            dfs1(v)

    # Passe 2 : DFS sur le graphe transposé, dans l'ordre décroissant de fin
    visited2 = [False] * n
    components = []

    def dfs2(v, comp):
        stack = [v]
        visited2[v] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for w in range(n):
                if P[w, node] > 1e-9 and not visited2[w]:
                    visited2[w] = True
                    stack.append(w)

    for v in reversed(finish_order):
        if not visited2[v]:
            comp = []
            dfs2(v, comp)
            components.append(comp)

    # Conversion indices → états, détermination du type
    classes = []
    class_type = {}
    class_of   = {}

    for ci, comp in enumerate(components):
        scc_set = {states[i] for i in comp}
        classes.append(scc_set)

        # Une SCC est récurrente si aucun arc ne sort vers l'extérieur
        is_closed = True
        for i in comp:
            for j in range(n):
                if P[i, j] > 1e-9 and j not in comp:
                    is_closed = False
                    break

        t = 'Récurrent' if is_closed else 'Transitoire'
        for i in comp:
            class_type[states[i]] = t
            class_of[states[i]]   = ci

    return classes, class_type, class_of


def absorption_analysis(P, states, state_idx):
    """
    — Analyse d'absorption 

    Décomposition canonique :
        P = [ I  0 ]   Q = sous-matrice états transitoires
            [ R  Q ]   R = transitions transitoires → absorbants

    Matrice fondamentale : N = (I - Q)^{-1}
      N[i,j] = nombre moyen de visites en j depuis i (avant absorption)

    Probabilités d'absorption : B = N · R
      B[i, k] = probabilité d'être absorbé par k en partant de i

    Temps moyen avant absorption : t̄_i = Σ_j N[i,j]

    Retourne
    --------
    dict avec clés : N, B, t_mean, transient_states, absorbing_states
    ou None si pas d'états absorbants/transitoires.
    """
    absorbing_set = {GOAL, FAIL}
    transient_states  = [s for s in states if s not in absorbing_set]
    absorbing_states  = [s for s in states if s     in absorbing_set]

    if not transient_states or not absorbing_states:
        return None

    t_idx = [state_idx[s] for s in transient_states]
    a_idx = [state_idx[s] for s in absorbing_states]

    Q = P[np.ix_(t_idx, t_idx)]   # transitions entre états transitoires
    R = P[np.ix_(t_idx, a_idx)]   # transitions vers états absorbants

    I = np.eye(len(transient_states))
    try:
        N     = np.linalg.inv(I - Q)   # matrice fondamentale
        B     = N @ R                   # probabilités d'absorption
        t_mean = N.sum(axis=1)          # temps moyen avant absorption
    except np.linalg.LinAlgError:
        return None

    return {
        'N'                : N,
        'B'                : B,
        't_mean'           : t_mean,
        'transient_states' : transient_states,
        'absorbing_states' : absorbing_states,
    }


# =============================================================================
# — Simulation Monte-Carlo
# =============================================================================

def simulate(policy, grid, start, epsilon=0.1, N=5000, max_steps=300, seed=42):
    """
    — Simule N trajectoires Markov à partir de start.

    À chaque étape, l'agent applique la politique avec incertitude ε :
      • action voulue       : prob 1 - ε
      • déviation latérale  : prob ε/2 chacune
      • collision → FAIL

    Paramètres
    ----------
    policy    : dict état → (action, état_suivant)
    grid      : grille 2D
    start     : état initial
    epsilon   : taux d'incertitude
    N         : nombre de trajectoires
    max_steps : pas maximum par trajectoire
    seed      : graine aléatoire pour la reproductibilité

    Retourne
    --------
    stats : dict contenant
        prob_goal   : Pb(atteindre GOAL)
        prob_fail   : Pb(atteindre FAIL / collision)
        prob_stuck  : Pb(dépasser max_steps sans absorption)
        mean_time   : E[temps d'atteinte GOAL | succès]
        std_time    : σ du temps d'atteinte GOAL
        times       : liste des temps d'atteinte (succès uniquement)
        sample_traj : quelques trajectoires pour visualisation
    """
    rng  = np.random.default_rng(seed)
    rows = len(grid)
    cols = len(grid[0])

    n_goal  = 0
    n_fail  = 0
    n_stuck = 0
    times   = []
    sample_traj = []        # 10 premières trajectoires

    for traj_id in range(N):
        state  = start
        record = traj_id < 10
        traj   = [state] if record else None
        outcome = None

        for step in range(max_steps):
            if state not in policy:
                n_stuck += 1
                outcome = 'stuck'
                break

            action, _ = policy[state]

            # ─ État but atteint ─────────────────────────────────────────
            if action == 'STAY':
                n_goal += 1
                times.append(step)
                outcome = 'goal'
                break

            # ─ Tirage stochastique de la direction effective ─────────────
            lats = LATERAL[action]
            r    = rng.random()
            if r < 1.0 - epsilon:
                chosen = action
            elif len(lats) == 2:
                chosen = lats[0] if r < 1.0 - epsilon / 2.0 else lats[1]
            else:
                chosen = action

            dr, dc = ACTION_DELTA[chosen]
            nr, nc = state[0] + dr, state[1] + dc

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                state = (nr, nc)
                if record:
                    traj.append(state)
            else:
                # Collision irréversible → FAIL
                n_fail += 1
                outcome = 'fail'
                break

        else:
            n_stuck += 1
            outcome = 'stuck'

        if record:
            sample_traj.append({'traj': traj, 'outcome': outcome})

    arr_times = np.array(times) if times else np.array([0])

    stats = {
        'prob_goal'  : n_goal  / N,
        'prob_fail'  : n_fail  / N,
        'prob_stuck' : n_stuck / N,
        'mean_time'  : float(arr_times.mean()) if len(times) > 0 else float('inf'),
        'std_time'   : float(arr_times.std())  if len(times) > 0 else 0.0,
        'times'      : arr_times.tolist(),
        'N'          : N,
        'sample_traj': sample_traj,
    }
    return stats


def compare_matrix_vs_simulation(history, states, state_idx, sim_stats, n_steps=60):
    """
    — Compare π^(n)[GOAL] (calcul matriciel) avec la distribution
            empirique cumulée (simulation Monte-Carlo).

    Retourne
    --------
    dict avec :
        t              : tableau d'indices temporels
        matrix_goal    : π^(n)[GOAL] pour chaque n
        sim_cumul_goal : proportion empirique cumulée d'atteinte de GOAL à n
    """
    goal_idx = state_idx.get(GOAL, None)
    if goal_idx is None:
        return None

    t_arr        = np.arange(n_steps + 1)
    matrix_goal  = history[:n_steps + 1, goal_idx]

    # Distribution cumulée empirique
    times_raw  = sim_stats['times']
    N          = sim_stats['N']
    cumul      = np.zeros(n_steps + 1)
    for t in times_raw:
        t_int = int(t)
        if t_int <= n_steps:
            cumul[t_int] += 1
    cumul = np.cumsum(cumul) / N

    return {
        't'             : t_arr,
        'matrix_goal'   : matrix_goal,
        'sim_cumul_goal': cumul,
    }