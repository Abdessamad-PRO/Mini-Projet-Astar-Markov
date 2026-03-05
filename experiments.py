import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── Modules du projet ─────────────────────────────────────────────────────────
from astar  import (neighbors, astar, ucs, greedy,
                    astar_manhattan, weighted_astar, extract_policy, manhattan)
from markov import (GOAL, FAIL,
                    build_transition_matrix, compute_pi_n,
                    build_transition_graph, find_communication_classes,
                    absorption_analysis, simulate,
                    compare_matrix_vs_simulation)

os.makedirs('figures', exist_ok=True)

# =============================================================================
# Grilles (— Définition du cas d'usage)
# =============================================================================

import numpy as np

# ── Grille 1 : Facile (8×8) ───────────────────────────────────────────────────
GRID_EASY = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)
START_EASY = (0, 0)
GOAL_EASY  = (7, 7)

# ── Grille 2 : Moyenne (12×12) ────────────────────────────────────────────────
GRID_MEDIUM = np.array([
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)
START_MEDIUM = (0, 0)
GOAL_MEDIUM  = (11, 11)

# ── Grille 3 : Difficile (16×16) ─────────────────────────────────────────────
def _build_hard_grid():
    g = np.zeros((16, 16), dtype=int)
    obs = [
        (0,3),(1,3),(2,3),(4,3),(5,3),(6,3),
        (0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),
        (0,11),(1,11),(2,11),(3,11),(4,11),(5,11),(6,11),
        (9,0),(9,1),(9,2),(9,4),(9,5),(9,6),(9,7),(9,8),
        (12,8),(12,9),(12,10),(12,11),(12,12),(12,13),(12,14),(12,15),
        (2,5),(2,6),(5,9),(5,10),(8,12),(8,13),
        (10,4),(10,5),(11,1),(11,2),(13,3),(14,6),(15,10),
    ]
    for r, c in obs:
        g[r, c] = 1
    return g

GRID_HARD  = _build_hard_grid()
START_HARD = (0, 0)
GOAL_HARD  = (15, 15)

GRIDS = {
    'Facile (8×8)'      : (GRID_EASY,   START_EASY,   GOAL_EASY),
    'Moyenne (12×12)'   : (GRID_MEDIUM, START_MEDIUM, GOAL_MEDIUM),
    'Difficile (16×16)' : (GRID_HARD,   START_HARD,   GOAL_HARD),
}

# =============================================================================
# Palette graphique
# =============================================================================
C = {
    'bg'     : '#F8F9FA',
    'wall'   : '#2C3E50',
    'free'   : '#FFFFFF',
    'start'  : '#E67E22',
    'goal'   : '#8E44AD',
    'ucs'    : '#2980B9',
    'greedy' : '#E74C3C',
    'astar'  : '#27AE60',
    'markov' : '#1ABC9C',
    'sim'    : '#E67E22',
    'visited': '#D5E8F5',
}

# =============================================================================
# Utilitaire : dessin de grille
# =============================================================================

def draw_grid(ax, grid, path=None, visited=None,
              start=None, goal=None, path_color='#27AE60',
              title='', show_legend=True):
    """Dessine la grille avec obstacles, chemin, start et goal."""
    rows, cols = grid.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_facecolor(C['bg'])

    # Cellules
    for r in range(rows):
        for c in range(cols):
            color = C['wall'] if grid[r, c] == 1 else C['free']
            rect = mpatches.FancyBboxPatch(
                (c + 0.04, rows - r - 1 + 0.04), 0.92, 0.92,
                boxstyle='round,pad=0.02', linewidth=0.4,
                edgecolor='#CCCCCC', facecolor=color)
            ax.add_patch(rect)

    # Nœuds visités
    if visited:
        for cell in visited:
            r, c = cell
            rect = mpatches.FancyBboxPatch(
                (c + 0.04, rows - r - 1 + 0.04), 0.92, 0.92,
                boxstyle='round,pad=0.02', linewidth=0,
                facecolor=C['visited'], alpha=0.6)
            ax.add_patch(rect)

    # Chemin (trait + flèches)
    if path and len(path) > 1:
        xs = [c + 0.5 for (r, c) in path]
        ys = [rows - r - 0.5 for (r, c) in path]
        ax.plot(xs, ys, '-', color=path_color, lw=2.2, zorder=3, alpha=0.9)
        for i in range(0, len(path) - 1, max(1, len(path) // 6)):
            r0, c0 = path[i]
            r1, c1 = path[i + 1]
            ax.annotate('', xy=(c1 + 0.5, rows - r1 - 0.5),
                        xytext=(c0 + 0.5, rows - r0 - 0.5),
                        arrowprops=dict(arrowstyle='-|>',
                                        color=path_color, lw=1.4,
                                        mutation_scale=10), zorder=4)

    # Start
    if start:
        r, c = start
        ax.plot(c + 0.5, rows - r - 0.5, 's', ms=13, color=C['start'], zorder=5)
        ax.text(c + 0.5, rows - r - 0.5, 'S', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=6)

    # Goal
    if goal:
        r, c = goal
        ax.plot(c + 0.5, rows - r - 0.5, '*', ms=17, color=C['goal'], zorder=5)
        ax.text(c + 0.5, rows - r - 0.5, 'G', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=6)

    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='#E0E0E0', linewidth=0.4)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=5)

def _print_table(header, rows, widths=None):
    """Affiche un tableau formaté dans la console."""
    if widths is None:
        widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(header)]
    sep  = '─' * (sum(widths) + 3 * len(widths) + 1)
    fmt  = ' │ '.join('{:<' + str(w) + '}' for w in widths)
    print(sep)
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))
    print(sep)


# =============================================================================
# E.1 — Comparer UCS / Greedy / A* sur 3 grilles 
# =============================================================================

def experiment_E1():
    print('\n' + '═' * 60)
    print('E.1 — Comparaison UCS / Greedy / A* sur 3 grilles')
    print('═' * 60)

    algos = [
        ('UCS',    ucs,           C['ucs']),
        ('Greedy', greedy,        C['greedy']),
        ('A*',     astar_manhattan, C['astar']),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('E.1 — Comparaison UCS / Greedy / A* sur 3 grilles',
                 fontsize=14, fontweight='bold', y=0.99)

    table_rows = []

    for row_i, (gname, (grid, start, goal)) in enumerate(GRIDS.items()):
        for col_i, (aname, afn, acol) in enumerate(algos):
            path, cost, expanded, open_max, elapsed, stats = afn(grid, start, goal)
            ax = axes[row_i][col_i]
            cost_str = f'{cost:.0f}' if cost != float('inf') else '∞'
            draw_grid(ax, grid, path=path, start=start, goal=goal,
                      path_color=acol,
                      title=(f'{aname} — {gname}\n'
                             f'Coût={cost_str}  Nœuds={expanded}  '
                             f'{elapsed*1000:.1f}ms'))
            table_rows.append([
                gname, aname, cost_str,
                str(expanded), str(open_max), f'{elapsed*1000:.2f}'
            ])
            print(f'  {gname:22s} | {aname:8s} | Coût={cost_str:5s} | '
                  f'Nœuds={expanded:4d} | Temps={elapsed*1000:.2f}ms')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('figures/E1_grilles.png', dpi=140, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()

    # ── Bar chart comparatif ──────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.patch.set_facecolor(C['bg'])
    fig2.suptitle('E.1 — Métriques comparatives : UCS / Greedy / A*',
                  fontsize=13, fontweight='bold')

    metrics  = ['path_cost', 'expanded', 'elapsed_ms']
    ylabels  = ['Coût du chemin', 'Nœuds développés', 'Temps CPU (ms)']
    anames_l = ['UCS', 'Greedy', 'A*']
    acolors  = [C['ucs'], C['greedy'], C['astar']]

    # Re-run pour collecter stats structurées
    all_stats = {gn: {} for gn in GRIDS}
    for gname, (grid, start, goal) in GRIDS.items():
        for aname, afn, _ in algos:
            _, cost, expanded, open_max, elapsed, stats = afn(grid, start, goal)
            if cost == float('inf'):
                stats['path_cost'] = 999
            all_stats[gname][aname] = stats

    x = np.arange(len(GRIDS))
    w = 0.25
    for mi, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes2[mi]
        ax.set_facecolor(C['bg'])
        for ai, (aname, acol) in enumerate(zip(anames_l, acolors)):
            vals = [all_stats[gn][aname].get(metric, 0) for gn in GRIDS]
            bars = ax.bar(x + ai * w, vals, w, label=aname,
                          color=acol, alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + w / 2,
                            bar.get_height() + max(vals) * 0.02,
                            f'{v:.0f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x + w)
        ax.set_xticklabels(list(GRIDS.keys()), rotation=12, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/E1_barchart.png', dpi=140, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()

    print('\n  Tableau E.1 :')
    _print_table(
        ['Grille', 'Algo', 'Coût', 'Nœuds', 'OPEN max', 'Temps (ms)'],
        table_rows,
        widths=[22, 8, 5, 6, 8, 10]
    )
    print('  → figures/E1_grilles.png')
    print('  → figures/E1_barchart.png')


# =============================================================================
# E.2 — Fixer A* et varier ε ∈ {0, 0.1, 0.2, 0.3}
#        Mesurer : (i) coût prévu (A*)  (ii) Pb(atteindre GOAL) Markov
# =============================================================================

def experiment_E2():
    print('\n' + '═' * 60)
    print('E.2 — Impact de ε sur le plan A* (Markov + Monte-Carlo)')
    print('═' * 60)

    grid, start, goal = GRIDS['Facile (8×8)']
    path, cost, _, _, _, _ = astar_manhattan(grid, start, goal)
    policy = extract_policy(path)
    n_steps = 60

    epsilons = [0.0, 0.1, 0.2, 0.3]
    palette  = ['#2ECC71', '#3498DB', '#E67E22', '#E74C3C']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle(
        f'E.2 — Impact de ε (Grille Facile, Chemin A* coût={cost:.0f})',
        fontsize=13, fontweight='bold', y=0.99)

    summary_ax = axes[1][1]
    summary_ax.set_facecolor(C['bg'])

    table_rows = []

    for ei, (eps, col) in enumerate(zip(epsilons, palette)):
        # ── Construction P, π^(n) ────────────────────────────────────────────
        P, states, sidx = build_transition_matrix(grid, policy, epsilon=eps)
        pi0 = np.zeros(len(states))
        pi0[sidx[start]] = 1.0
        history = compute_pi_n(pi0, P, n_steps)

        # ── Simulation Monte-Carlo ───────────────────────────────────────────
        sim = simulate(policy, grid, start, epsilon=eps, N=4000, seed=42)

        # ── Comparaison matriciel vs simulation ──────────────────────────────
        comp = compare_matrix_vs_simulation(history, states, sidx, sim, n_steps)

        row_i, col_i = ei // 2, ei % 2
        ax = axes[row_i][col_i]
        ax.set_facecolor(C['bg'])

        if comp is not None:
            t = comp['t']
            ax.plot(t, comp['matrix_goal'], '-',
                    color=C['markov'], lw=2.5, label='π^(n)[GOAL] matriciel')
            ax.plot(t, comp['sim_cumul_goal'], '--',
                    color=C['sim'], lw=2.0, label='Monte-Carlo cumulé')
            ax.fill_between(t, comp['matrix_goal'], comp['sim_cumul_goal'],
                            alpha=0.12, color='gray')

        ax.set_title(f'ε = {eps}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Étapes n', fontsize=9)
        ax.set_ylabel('Probabilité', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        # Courbe synthétique
        if comp is not None:
            summary_ax.plot(comp['t'], comp['matrix_goal'],
                            color=col, lw=2.2, label=f'ε={eps}')

        t_mean_str = (f"{sim['mean_time']:.1f}"
                      if sim['mean_time'] != float('inf') else '∞')
        table_rows.append([
            f'ε = {eps}', f'{cost:.0f}',
            f"{sim['prob_goal']:.4f}", f"{sim['prob_fail']:.4f}",
            t_mean_str
        ])
        print(f'  ε={eps:.1f} | Coût A*={cost:.0f} | '
              f"Pb(GOAL)={sim['prob_goal']:.4f} | "
              f"Pb(FAIL)={sim['prob_fail']:.4f} | "
              f"T_moy={t_mean_str}")

    summary_ax.set_title('Synthèse : π^(n)[GOAL] pour différents ε',
                          fontsize=10, fontweight='bold')
    summary_ax.set_xlabel('Étapes n', fontsize=9)
    summary_ax.set_ylabel('P(GOAL)', fontsize=9)
    summary_ax.set_ylim(0, 1.05)
    summary_ax.legend(fontsize=8)
    summary_ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('figures/E2_epsilon.png', dpi=140, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()

    print('\n  Tableau E.2 :')
    _print_table(
        ['ε', 'Coût A*', 'Pb(GOAL)', 'Pb(FAIL)', 'T_moy GOAL'],
        table_rows,
        widths=[8, 8, 10, 10, 12]
    )
    print('  → figures/E2_epsilon.png')


# =============================================================================
# E.3 — Comparer deux heuristiques admissibles : h=0 vs h=Manhattan
# =============================================================================

def experiment_E3():
    print('\n' + '═' * 60)
    print('E.3 — Comparaison heuristiques : h=0 vs h=Manhattan')
    print('═' * 60)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('E.3 — h=0 (UCS) vs h=Manhattan : même optimalité, moins de nœuds',
                 fontsize=13, fontweight='bold', y=0.99)

    table_rows = []

    for gi, (gname, (grid, start, goal)) in enumerate(GRIDS.items()):
        # h = 0
        path0, cost0, exp0, om0, el0, _ = ucs(grid, start, goal)
        cost0_s = f'{cost0:.0f}' if cost0 != float('inf') else '∞'

        # h = Manhattan
        pathM, costM, expM, omM, elM, _ = astar_manhattan(grid, start, goal)
        costM_s = f'{costM:.0f}' if costM != float('inf') else '∞'

        reduction = (f'{(1 - expM/exp0)*100:.0f}%'
                     if exp0 > 0 and expM != exp0 else '0%')

        draw_grid(axes[0][gi], grid, path=path0, start=start, goal=goal,
                  path_color=C['ucs'],
                  title=(f'h=0 (UCS) — {gname}\n'
                         f'Coût={cost0_s}  Nœuds={exp0}'))
        draw_grid(axes[1][gi], grid, path=pathM, start=start, goal=goal,
                  path_color=C['astar'],
                  title=(f'h=Manhattan — {gname}\n'
                         f'Coût={costM_s}  Nœuds={expM}  (−{reduction})'))

        table_rows.append([gname, 'h=0',        cost0_s, str(exp0), str(om0), f'{el0*1000:.2f}', '—'])
        table_rows.append([gname, 'h=Manhattan', costM_s, str(expM), str(omM), f'{elM*1000:.2f}', f'−{reduction}'])
        print(f'  {gname:22s} | h=0        : Nœuds={exp0:4d} Coût={cost0_s}')
        print(f'  {gname:22s} | Manhattan  : Nœuds={expM:4d} Coût={costM_s} (réduction {reduction})')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('figures/E3_heuristics.png', dpi=140, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()

    print('\n  Tableau E.3 :')
    _print_table(
        ['Grille', 'Heuristique', 'Coût', 'Nœuds', 'OPEN max', 'Temps (ms)', 'Réduction'],
        table_rows,
        widths=[22, 12, 5, 6, 8, 10, 9]
    )
    print('  → figures/E3_heuristics.png')


# =============================================================================
# E.4 — (Option) Weighted A* — compromis vitesse / optimalité
# =============================================================================

def experiment_E4():
    print('\n' + '═' * 60)
    print('E.4 — Weighted A* : compromis vitesse / optimalité')
    print('═' * 60)

    weights = [1.0, 1.5, 2.0, 3.0, 5.0]
    table_rows = []

    fig, axes = plt.subplots(1, len(weights), figsize=(4 * len(weights), 5))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('E.4 — Weighted A* (w=1.0 → 5.0) sur grille Difficile',
                 fontsize=13, fontweight='bold', y=1.01)

    grid, start, goal = GRIDS['Difficile (16×16)']
    opt_cost = None
    cmap = plt.cm.RdYlGn_r

    for wi, w in enumerate(weights):
        path, cost, expanded, open_max, elapsed, stats = weighted_astar(
            grid, start, goal, w=w)
        if w == 1.0:
            opt_cost = cost

        cost_s = f'{cost:.0f}' if cost != float('inf') else '∞'
        ratio  = (f'{cost/opt_cost:.3f}'
                  if opt_cost and opt_cost != float('inf') else '—')

        col = cmap(wi / (len(weights) - 1))
        draw_grid(axes[wi], grid, path=path, start=start, goal=goal,
                  path_color=col,
                  title=(f'w={w}\nCoût={cost_s}\nNœuds={expanded}'))

        table_rows.append([f'w={w}', cost_s, str(expanded), ratio, f'{elapsed*1000:.2f}'])
        print(f'  w={w} | Coût={cost_s} | Nœuds={expanded} | '
              f'Ratio={ratio} | {elapsed*1000:.2f}ms')

    plt.tight_layout()
    plt.savefig('figures/E4_weighted.png', dpi=140, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()

    # ── Courbes coût & nœuds vs w ─────────────────────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.patch.set_facecolor(C['bg'])
    fig2.suptitle('E.4 — Impact de w sur coût et nœuds développés',
                  fontsize=12, fontweight='bold')

    costs_l = [float(r[1]) if r[1] != '∞' else 999 for r in table_rows]
    nodes_l = [int(r[2]) for r in table_rows]

    ax1.plot([str(w) for w in weights], costs_l, 'o-',
             color=C['astar'], lw=2.5, ms=9)
    ax1.axhline(opt_cost or 0, color='gray', ls='--', alpha=0.6, label='Optimal')
    ax1.set_title('Coût du chemin vs w', fontweight='bold')
    ax1.set_xlabel('Poids w'); ax1.set_ylabel('Coût')
    ax1.legend(); ax1.spines[['top', 'right']].set_visible(False)

    ax2.bar([str(w) for w in weights], nodes_l,
            color=C['ucs'], alpha=0.85, edgecolor='white')
    ax2.set_title('Nœuds développés vs w', fontweight='bold')
    ax2.set_xlabel('Poids w'); ax2.set_ylabel('Nœuds')
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/E4_weighted_curves.png', dpi=140, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()

    print('\n  Tableau E.4 :')
    _print_table(
        ['w', 'Coût', 'Nœuds', 'Ratio/optimal', 'Temps (ms)'],
        table_rows,
        widths=[6, 5, 6, 13, 10]
    )
    print('  → figures/E4_weighted.png')
    print('  → figures/E4_weighted_curves.png')


# =============================================================================
# Analyse Markov complète (phases P3/P4/P5) — figure de synthèse
# =============================================================================

def experiment_markov_full():
    print('\n' + '═' * 60)
    print('Analyse Markov complète (P3 + P4 + P5)')
    print('═' * 60)

    grid, start, goal = GRIDS['Facile (8×8)']
    path, cost, _, _, _, _ = astar_manhattan(grid, start, goal)
    policy = extract_policy(path)
    eps    = 0.15
    n_steps = 70

    # P3 — Matrice P et π^(n)
    P, states, sidx = build_transition_matrix(grid, policy, epsilon=eps)
    pi0 = np.zeros(len(states))
    pi0[sidx[start]] = 1.0
    history = compute_pi_n(pi0, P, n_steps)

    # P4 — Classes et absorption
    classes, class_type, class_of = find_communication_classes(P, states)
    absorp = absorption_analysis(P, states, sidx)

    # P5 — Simulation
    sim  = simulate(policy, grid, start, epsilon=eps, N=5000, seed=0)
    comp = compare_matrix_vs_simulation(history, states, sidx, sim, n_steps)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(C['bg'])
    gs = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    # A — Grille + chemin
    ax_g = fig.add_subplot(gs[0, 0])
    draw_grid(ax_g, grid, path=path, start=start, goal=goal,
              path_color=C['astar'],
              title=f'Chemin A* (ε={eps})\nCoût={cost:.0f}')

    # B — π^(n) pour états clés
    ax_pi = fig.add_subplot(gs[0, 1])
    ax_pi.set_facecolor(C['bg'])
    t_arr = np.arange(n_steps + 1)
    ax_pi.plot(t_arr, history[:, sidx[GOAL]], color=C['astar'],
               lw=2.5, label='π^(n)[GOAL]')
    ax_pi.plot(t_arr, history[:, sidx[FAIL]], color=C['greedy'],
               lw=2.0, ls='--', label='π^(n)[FAIL]')
    ax_pi.plot(t_arr, history[:, sidx[start]], color=C['ucs'],
               lw=1.5, ls=':', label=f'π^(n)[start]')
    ax_pi.set_title(f'Évolution π^(n) = π^(0)·P^n\n(ε={eps})',
                    fontsize=10, fontweight='bold')
    ax_pi.set_xlabel('Étapes n'); ax_pi.set_ylabel('Probabilité')
    ax_pi.set_ylim(0, 1.05); ax_pi.legend(fontsize=8)
    ax_pi.spines[['top', 'right']].set_visible(False)

    # C — Heatmap P
    ax_hm = fig.add_subplot(gs[0, 2])
    n_show = min(18, len(states))
    im = ax_hm.imshow(P[:n_show, :n_show], cmap='Blues', vmin=0, vmax=1)
    ax_hm.set_title(f'Matrice P (premiers {n_show} états)',
                    fontsize=10, fontweight='bold')
    ax_hm.set_xlabel('j'); ax_hm.set_ylabel('i')
    plt.colorbar(im, ax=ax_hm, shrink=0.8)

    # D — Distribution temps d'atteinte (simulation)
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist.set_facecolor(C['bg'])
    times_arr = sim['times']
    if times_arr:
        ax_hist.hist(times_arr, bins=min(30, len(set(int(t) for t in times_arr))),
                     color=C['sim'], edgecolor='white', alpha=0.85)
        ax_hist.axvline(sim['mean_time'], color='red', lw=2,
                        label=f"μ = {sim['mean_time']:.1f}")
    ax_hist.set_title(f'Distribution du temps → GOAL\n(N=5000, ε={eps})',
                      fontsize=10, fontweight='bold')
    ax_hist.set_xlabel('Étapes'); ax_hist.set_ylabel('Fréquence')
    ax_hist.legend(fontsize=8); ax_hist.spines[['top', 'right']].set_visible(False)

    # E — Comparaison matriciel vs Monte-Carlo
    ax_cmp = fig.add_subplot(gs[1, 1])
    ax_cmp.set_facecolor(C['bg'])
    if comp:
        ax_cmp.plot(comp['t'], comp['matrix_goal'], '-',
                    color=C['markov'], lw=2.5, label='π^(n)[GOAL] (matriciel)')
        ax_cmp.plot(comp['t'], comp['sim_cumul_goal'], '--',
                    color=C['sim'],    lw=2.0, label='Monte-Carlo (cumulé)')
        ax_cmp.fill_between(comp['t'], comp['matrix_goal'],
                             comp['sim_cumul_goal'],
                             alpha=0.15, color='gray', label='Écart')
    ax_cmp.set_title('Markov matriciel vs Monte-Carlo',
                     fontsize=10, fontweight='bold')
    ax_cmp.set_xlabel('Étapes n'); ax_cmp.set_ylabel('P(GOAL)')
    ax_cmp.set_ylim(0, 1.05); ax_cmp.legend(fontsize=8)
    ax_cmp.spines[['top', 'right']].set_visible(False)

    # F — Résumé absorption
    ax_abs = fig.add_subplot(gs[1, 2])
    ax_abs.set_facecolor(C['bg'])
    ax_abs.axis('off')

    lines = [f'Analyse d\'absorption (ε={eps})', '']
    lines += [f'Simulation Monte-Carlo (N=5000) :',
              f"  Pb(GOAL)  = {sim['prob_goal']:.4f}",
              f"  Pb(FAIL)  = {sim['prob_fail']:.4f}",
              f"  Pb(stuck) = {sim['prob_stuck']:.4f}",
              f"  T_moyen   = {sim['mean_time']:.2f} étapes",
              f"  σ_T       = {sim['std_time']:.2f}", '']

    if absorp:
        abs_states = absorp['absorbing_states']
        tr_states  = absorp['transient_states']
        if GOAL in abs_states and start in tr_states:
            g_col  = abs_states.index(GOAL)
            s_row  = tr_states.index(start)
            lines += ['Calcul matriciel N=(I-Q)⁻¹ :',
                      f"  Pb(GOAL|start) = {absorp['B'][s_row,g_col]:.4f}",
                      f"  T_moyen (fond) = {absorp['t_mean'][s_row]:.2f}"]

    n_tr = sum(1 for s in states if class_type.get(s) == 'Transitoire')
    n_rc = sum(1 for s in states if class_type.get(s) == 'Récurrent')
    lines += ['', f'Classes :',
              f'  Transitoires : {n_tr}',
              f'  Récurrentes  : {n_rc}']

    ax_abs.text(0.05, 0.95, '\n'.join(lines),
                transform=ax_abs.transAxes,
                fontsize=8.5, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#EAF2FB',
                          edgecolor='#2980B9', alpha=0.85))
    ax_abs.set_title('Résumé Markov', fontsize=10, fontweight='bold')

    fig.suptitle(f'Analyse Markov complète — Grille Facile, ε={eps}',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('figures/Markov_full.png', dpi=140, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()

    print(f"  Pb(GOAL) simulation : {sim['prob_goal']:.4f}")
    print(f"  Pb(FAIL) simulation : {sim['prob_fail']:.4f}")
    if absorp:
        abs_states = absorp['absorbing_states']
        tr_states  = absorp['transient_states']
        if GOAL in abs_states and start in tr_states:
            g_col = abs_states.index(GOAL)
            s_row = tr_states.index(start)
            print(f"  Pb(GOAL) matriciel  : {absorp['B'][s_row,g_col]:.4f}")
            print(f"  T_moyen matriciel   : {absorp['t_mean'][s_row]:.2f}")
    print('  → figures/Markov_full.png')

def draw_transition_graph(ax, P, states, state_idx, grid, path,
                          start, goal, epsilon, class_type,
                          threshold=0.01, title=''):
    """
    Dessine le graphe orienté des transitions de la chaîne de Markov.
    Positions des noeuds calées sur la grille. Probabilités sur chaque arc.
    Couleurs : Start=orange, GOAL=violet, FAIL=rouge, chemin=vert, autre=gris.
    Épaisseur des arcs proportionnelle à la probabilité.
    """
    rows_g = len(grid)
    cols_g = len(grid[0])

    # ── Positions des noeuds ─────────────────────────────────────────────────
    pos = {}
    for s in states:
        if s == 'GOAL':
            pos[s] = (cols_g + 1.5,  rows_g / 2)
        elif s == 'FAIL':
            pos[s] = (cols_g + 1.5,  rows_g / 2 - 2.5)
        else:
            r, c = s
            pos[s] = (c, rows_g - 1 - r)

    path_set = set(path) if path else set()

    # ── Couleurs des noeuds ───────────────────────────────────────────────────
    node_colors = {}
    for s in states:
        if s == start:
            node_colors[s] = C['start']
        elif s == 'GOAL':
            node_colors[s] = C['goal']
        elif s == 'FAIL':
            node_colors[s] = C['greedy']
        elif s in path_set:
            node_colors[s] = '#A9DFBF'
        else:
            node_colors[s] = '#D5D8DC'

    n = len(states)
    ax.set_facecolor(C['bg'])

    # ── Arcs ─────────────────────────────────────────────────────────────────
    for i, si in enumerate(states):
        xi, yi = pos[si]
        for j, sj in enumerate(states):
            pij = P[i, j]
            if pij < threshold:
                continue
            xj, yj = pos[sj]
            lw    = 0.6 + 3.5 * pij
            alpha = 0.35 + 0.65 * pij

            if i == j:
                # Boucle (auto-transition)
                loop_r = 0.38
                theta  = np.linspace(0, 2 * np.pi, 120)
                lx = xi + loop_r * np.cos(theta) + loop_r * 0.6
                ly = yi + loop_r * np.sin(theta) + loop_r * 0.6
                ax.plot(lx, ly, '-', color='#7F8C8D', lw=lw * 0.7,
                        alpha=alpha, zorder=2)
                ax.annotate('', xy=(lx[30], ly[30]),
                            xytext=(lx[25], ly[25]),
                            arrowprops=dict(arrowstyle='-|>',
                                            color='#7F8C8D',
                                            lw=0.8, mutation_scale=7),
                            zorder=3)
                mx, my = xi + loop_r * 1.15 + 0.05, yi + loop_r * 1.6
                ax.text(mx, my, f'{pij:.2f}', fontsize=5.5,
                        ha='center', va='center', color='#5D6D7E',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.12',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.85))
            else:
                # Arc orienté
                dx, dy = xj - xi, yj - yi
                norm   = max((dx**2 + dy**2)**0.5, 1e-9)
                perp_x, perp_y = -dy / norm * 0.18, dx / norm * 0.18
                r_node = 0.28
                sx_ = xi + dx / norm * r_node + perp_x
                sy_ = yi + dy / norm * r_node + perp_y
                ex_ = xj - dx / norm * r_node + perp_x
                ey_ = yj - dy / norm * r_node + perp_y

                # Couleur selon destination
                if sj == 'GOAL':
                    arc_color = C['astar']
                elif sj == 'FAIL':
                    arc_color = C['greedy']
                elif pij >= 1.0 - epsilon - 0.01:
                    arc_color = '#2980B9'
                else:
                    arc_color = '#E67E22'

                ax.annotate('',
                            xy=(ex_, ey_), xytext=(sx_, sy_),
                            arrowprops=dict(
                                arrowstyle='-|>',
                                color=arc_color, lw=lw,
                                connectionstyle='arc3,rad=0.15',
                                mutation_scale=9),
                            zorder=2, alpha=alpha)

                # Probabilité sur l'arc
                mx = (sx_ + ex_) / 2 + perp_x * 0.6
                my = (sy_ + ey_) / 2 + perp_y * 0.6
                ax.text(mx, my, f'{pij:.2f}', fontsize=5.5,
                        ha='center', va='center', color='#2C3E50',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.12',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.85),
                        zorder=4)

    # ── Noeuds ───────────────────────────────────────────────────────────────
    for s in states:
        x, y = pos[s]
        circle = plt.Circle((x, y), 0.28,
                             color=node_colors[s],
                             ec='#2C3E50', lw=1.0, zorder=5)
        ax.add_patch(circle)
        if s == 'GOAL':
            label, fs, fc = 'GOAL', 6, 'white'
        elif s == 'FAIL':
            label, fs, fc = 'FAIL', 6, 'white'
        elif s == start:
            label, fs, fc = 'S\n' + str(s), 5, 'white'
        else:
            label, fs, fc = str(s), 5, '#2C3E50'
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=fc, zorder=6)

    # ── Legende ──────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C['start'],  label='Start'),
        mpatches.Patch(color=C['goal'],   label='GOAL (absorbant)'),
        mpatches.Patch(color=C['greedy'], label='FAIL (absorbant)'),
        mpatches.Patch(color='#A9DFBF',   label='Transitoire (chemin)'),
        mpatches.Patch(color='#D5D8DC',   label='Transitoire (hors chemin)'),
    ]
    ax.legend(handles=legend_items, fontsize=6, loc='lower left',
              framealpha=0.85, edgecolor='#CCCCCC')

    ax.set_xlim(-0.8, cols_g + 3.2)
    ax.set_ylim(-1.0, rows_g + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)


def experiment_transition_graph():
    """
    - Affichage du graphe de transitions pour 3 valeurs de epsilon.
    Montre comment l'incertitude cree des arcs de deviation laterale
    et des transitions vers l'etat FAIL.
    """
    print('\n' + '=' * 60)
    print('P4.1 - Graphe de transitions (probabilites sur les arcs)')
    print('=' * 60)

    grid, start, goal = GRIDS['Facile (8×8)']
    if grid is None:
        # fallback : premiere grille
        gname = list(GRIDS.keys())[0]
        grid, start, goal = GRIDS[gname]

    path, cost, _, _, _, _ = astar_manhattan(grid, start, goal)
    policy = extract_policy(path)
    epsilons_g = [0.0, 0.1, 0.2]

    fig, axes = plt.subplots(1, 3, figsize=(21, 9))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle(
        'P4.1 - Graphe de transitions de la chaine de Markov\n'
        '(bleu = action principale, orange = deviation, '
        'vert = vers GOAL, rouge = vers FAIL)',
        fontsize=12, fontweight='bold', y=1.01)

    for ax, eps in zip(axes, epsilons_g):
        P, states, sidx = build_transition_matrix(grid, policy, epsilon=eps)
        classes, class_type, _ = find_communication_classes(P, states)
        n_arcs = int((P > 0.01).sum())
        draw_transition_graph(
            ax, P, states, sidx, grid, path, start, goal,
            epsilon=eps, class_type=class_type,
            threshold=0.01,
            title=(f'epsilon = {eps}\n'
                   f'{len(states)} etats  {n_arcs} arcs (prob > 0.01)\n'
                   f'Transitoires : {sum(1 for t in class_type.values() if t=="Transitoire")}  '
                   f'Recurrents : {sum(1 for t in class_type.values() if t=="Récurrent")}')
        )
        print(f'  epsilon={eps} : {len(states)} etats, {n_arcs} arcs')

    plt.tight_layout()
    plt.savefig('figures/Transition_graph.png', dpi=140, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()
    print('  -> figures/Transition_graph.png')

    # Vue zoomee sur les etats du chemin uniquement (epsilon=0.1)
    eps = 0.1
    P, states, sidx = build_transition_matrix(grid, policy, epsilon=eps)
    classes, class_type, _ = find_communication_classes(P, states)

    path_states = path + ['GOAL', 'FAIL']
    path_idx    = [sidx[s] for s in path_states]
    P_sub       = P[np.ix_(path_idx, path_idx)].copy()
    sub_sidx    = {s: i for i, s in enumerate(path_states)}
    sub_ctype   = {s: class_type.get(s, 'Transitoire') for s in path_states}

    for i in range(len(path_states)):
        rs = P_sub[i].sum()
        if rs > 0:
            P_sub[i] /= rs

    fig2, ax2 = plt.subplots(figsize=(18, 8))
    fig2.patch.set_facecolor(C['bg'])
    draw_transition_graph(
        ax2, P_sub, path_states, sub_sidx, grid, path,
        start, goal, epsilon=eps, class_type=sub_ctype,
        threshold=0.005,
        title=(f'Graphe de transitions - Etats du chemin A* uniquement (epsilon={eps})\n'
               f'Chaque arc affiche P(i->j) avec deviation laterale epsilon/2={eps/2}')
    )
    plt.tight_layout()
    plt.savefig('figures/Transition_graph_zoom.png', dpi=150, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()
    print('  -> figures/Transition_graph_zoom.png')
   
# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print()
    print(' Planification robuste sur grille : A* + Chaînes de Markov')

    experiment_E1()
    experiment_E2()
    experiment_E3()
    experiment_E4()
    experiment_markov_full()
    experiment_transition_graph()

    print('\n' + '=' * 60)
    print('  Toutes les figures générées dans figures/')
    figs = sorted(os.listdir('figures'))
    for f in figs:
        print(f'    {f}')