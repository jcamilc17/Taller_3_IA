"""Punto 1. Distribuciones discretas y decisiones de respuesta."""
import math


def normalize(weights: dict) -> dict:
    """Retorna un diccionario nuevo cuyas probabilidades suman uno.

    Entrada: weights = {resultado: peso}.
    Salida: {resultado: probabilidad}, con las mismas claves.

    Tips:
    - Conserve las claves y su orden; no modifique weights.
    - Rechace pesos negativos, no finitos, diccionarios vacíos y suma cero.
    """
    # Validacion de la entrada, weights no debe ser vacio.
    if not weights:
        raise ValueError("La distribución no puede estar vacía")
        
    for key, w in weights.items():
        # Validacion de la entrada, los pesos deben ser numeros finitos, no booleanos y no negativos.
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
            raise ValueError(f"Peso inválido para {key!r}: {w}")
            
    total = sum(weights.values())
    
    # Validacion de la entrada, la suma de los pesos no debe ser cero.
    if total == 0:
        raise ValueError("La suma de los pesos es cero")
        
    # Retorno del nuevo diccionario con las probabilidades calculadas.
    return {key: w / total for key, w in weights.items()}


def expected_utility(distribution: dict, utilities: dict) -> float:
    """Calcula sum_s P(s) U(s) para una acción.

    Entradas: distribution = {resultado: probabilidad} y
    utilities = {resultado: utilidad} para una sola acción.
    Salida: un número con la utilidad esperada.

    Tips:
    - distribution ya debe estar normalizada; no normalice utilidades.
    - Ambas tablas deben contener los mismos resultados posibles.
    """
    # Validacion de la entrada, las probabilidades y utilidades deben tener las mismas claves y no estar vacías.
    if set(distribution) != set(utilities) or not distribution:
        raise ValueError("Probabilidades y utilidades deben tener las mismas claves")
        
    # Validacion de la entrada, la distribución no debe contener valores negativos o no finitos.
    if any(not math.isfinite(p) or p < 0 for p in distribution.values()):
        raise ValueError("Distribución inválida")
        
    # Validacion de la entrada, la suma de las probabilidades de la distribución debe ser uno.
    if not math.isclose(sum(distribution.values()), 1, abs_tol=1e-9):
        raise ValueError("La distribución debe sumar uno")
        
    # Validacion de la entrada, las utilidades deben ser valores finitos.
    if any(not math.isfinite(u) for u in utilities.values()):
        raise ValueError("Utilidades no finitas")
    
    # Retorno de la utilidad esperada calculada mediante la suma ponderada.
    return sum(p * utilities[s] for s, p in distribution.items())


def best_action(distribution: dict, actions: dict) -> tuple[str, dict]:
    """Retorna (nombre de acción, utilidades esperadas por acción).

    Entradas: distribution = {resultado: probabilidad} y
    actions = {acción: {resultado: utilidad}}.
    Salida: (acción elegida, {acción: utilidad esperada}).

    Tips:
    - actions asocia cada acción con su tabla de utilidades.
    - Reutilice expected_utility y conserve la primera acción en los empates.
    """
    # Validacion de la entrada, se requiere al menos una acción en el diccionario.
    if not actions:
        raise ValueError("Se requiere al menos una acción")
        
    values = {}
    chosen = None
    for action, utilities in actions.items():
        values[action] = expected_utility(distribution, utilities)
        
        # Seleccion de la mejor acción, utilizando '>' estricto para conservar la primera en empates.
        if chosen is None or values[action] > values[chosen]:
            chosen = action
            
    # Retorno de la tupla con la acción elegida y el diccionario de utilidades esperadas.
    return chosen, values
