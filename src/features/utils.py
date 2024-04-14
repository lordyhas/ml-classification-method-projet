def find_best_split(size: int, expected: int = 5) -> tuple:
    """
    Trouve le meilleur diviseur pour effectuer une validation croisée sur un ensemble de données.

    Cette fonction identifie le diviseur optimal pour diviser un ensemble de données en sous-ensembles
    de taille égale pour la validation croisée, tout en s'assurant que le nombre de divisions
    ne dépasse pas le nombre attendu.

    Paramètres :
    size (int) : La taille totale de l'ensemble de données.
    expected (int, optionnel) : Le nombre maximal de divisions souhaité pour la validation croisée.
                                La valeur par défaut est 5.

    Retourne :
    tuple : Contient deux éléments :
            - Le dernier diviseur trouvé qui est inférieur ou égal à 'expected'.
            - Une liste de tous les diviseurs trouves qui sont inférieurs ou égaux à 'expected'.

    Exemple :
    >>> find_best_split(100)
    (5, [1, 2, 4, 5])
    """

    if size <= 0:
        raise ValueError("La taille doit être un entier positif non nul.")
    if expected <= 0:
        raise ValueError("Le nombre attendu de divisions doit être un entier positif non nul.")

    splits = []
    # Parcourt tous les entiers de 1 à 'size' pour trouver les diviseurs
    for i in range(1, size + 1):
        if size % i == 0:
            if i > expected:
                break
            splits.append(i)

    # Si aucun diviseur n'a été trouvé, retourne None
    if not splits:
        return None, []
    return splits[-1], splits
