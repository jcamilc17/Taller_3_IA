"""Punto 2. Inferencia exacta por enumeración."""
from algorithms.probability import normalize


def enumeration_ask(network, query: str, evidence: dict) -> dict:
    """Calcula P(query | evidence) sobre la conjunta factorizada.

    Entradas: network es la red; query es el nombre de la
    variable consultada; evidence es {nombre de variable: 0 o 1}.
    Salida: {0: probabilidad, 1: probabilidad}.

    Tips:
    - Recorra network.variables en orden topológico.
    - Para una variable fijada multiplique su probabilidad; para una oculta
      sume las dos alternativas 0 y 1. Normalice al terminar.
    - Use copias del evento y no modifique evidence.
    - network.probability(variable, valor, evento) consulta la CPT;
      el evento debe contener los valores de todos los padres.
    - Puede usar una función auxiliar recursiva. Al terminar el recorrido,
      el producto sin factores restantes vale 1.
    - Evidencia imposible: ValueError. Consulta observada: distribución puntual
      solo si la evidencia es posible.
    """
    # Valida nombres de variables y dominio de la evidencia.
    network.validate_query(query, evidence)
 
    variables = network.variables  # orden topológico
 
    def enumerate_all(index, event):
        # Validacion de la condicion de parada, si se han recorrido todas las variables, el producto sin factores restantes vale 1.
        if index == len(variables):
            return 1.0  # producto sin factores restantes
            
        variable = variables[index]
        
        # Validacion de la variable actual, si ya está fijada en la evidencia se multiplica su probabilidad.
        if variable in event:  # variable fijada: multiplicar
            return (network.probability(variable, event[variable], event)
                    * enumerate_all(index + 1, event))
                    
        total = 0.0  # variable oculta: sumar sobre 0 y 1
        for value in (0, 1):
            extended = {**event, variable: value}  # copia del evento
            total += (network.probability(variable, value, extended)
                      * enumerate_all(index + 1, extended))
                      
        # Retorno de la suma total calculada para la variable oculta sobre sus alternativas 0 y 1.
        return total
 
    # Validacion de la consulta, si la variable consultada ya fue observada en la evidencia.
    if query in evidence:  # consulta observada
        # Validacion de la evidencia, se comprueba si la probabilidad conjunta evaluada es cero para determinar si la evidencia es imposible.
        if enumerate_all(0, dict(evidence)) == 0:
            raise ValueError("La evidencia es imposible")
            
        # Retorno de la distribución puntual basada en el valor observado en la evidencia.
        return {0: float(evidence[query] == 0), 1: float(evidence[query] == 1)}
 
    weights = {value: enumerate_all(0, {**evidence, query: value}) for value in (0, 1)}
    
    # Validacion de los pesos resultantes, si la suma de los pesos es cero la evidencia es imposible.
    if weights[0] + weights[1] == 0:
        raise ValueError("La evidencia es imposible")
        
    # Retorno del diccionario normalizado con las probabilidades finales de la consulta.
    return normalize(weights)
