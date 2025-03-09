import random

def is_valid_lr_params(params, param_grid_lr):
    """Überprüft, ob die Hyperparameter für LogisticRegression gültig sind."""
    penalty, C, solver, max_iter = params

    solver_penalty_map = {
        "lbfgs": ["l2", None],
        "liblinear": ["l1", "l2"],
        "saga": ["l1", "l2", "elasticnet", None]
    }

    # Prüfen, ob der Solver den Penalty-Typ unterstützt
    if solver not in solver_penalty_map or penalty not in solver_penalty_map[solver]:
        return False  

    # Falls `penalty="elasticnet"`, muss `l1_ratio` explizit definiert sein
    if penalty == "elasticnet" and "l1_ratio" not in param_grid_lr:
        return False  

    return True

def select_diverse_combinations(param_combinations, max_combinations=20):
    """Wählt eine diverse Menge an Hyperparameter-Kombinationen aus."""
    if len(param_combinations) <= max_combinations:
        return param_combinations  # Falls bereits <= 20, einfach zurückgeben
    
    # Zufällige Auswahl mit gleichmäßiger Verteilung
    selected = set()
    while len(selected) < max_combinations:
        candidate = random.choice(param_combinations)
        selected.add(candidate)  # Set verhindert doppelte Einträge
    
    return list(selected)