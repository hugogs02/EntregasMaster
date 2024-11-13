import sys
from mrjob.job import MRJob
from collections import defaultdict

def procesaArchivo(line: str) -> tuple[str, str, float, float, float, float]:
    """
    Recibe una línea de un archivo de texto, con los campos separados por \
    comas, y devuelve los siguientes campos de interés: temporada, nombre del \
    jugador, puntos anotado, rebotes cogidos, asistencias dadas y net rating.

    Parameters
    ----------
        line : La línea del archivo.

    Returns
    ----------
        tuple[str, str, float, float, float, float] : Una tupla con los campos\
        de interés.
    """
    linea = line.split(',')
    temporada = linea[1]
    jugador = linea[0]
    pts = float(linea[3])
    reb = float(linea[4])
    ast = float(linea[5])
    net = float(linea[6])

    return temporada, jugador, pts, reb, ast, net

def calculaValor(p: float, r: float, a: float, n: float) -> float:
    """
    Función que recibe los datos necesarios para calcular la valoración de un \
    jugador (puntos, rebotes, asistencias y rating) y la calcula.

    Parameters
    ----------
        p : Los puntos anotados por el jugador.
        r : Los rebotes cogidos por el jugador.
        a : Las asistencias dadas por el jugador.
        n : El net rating del jugador.

    Returns
    ----------
        float : La valoración de dicho jugador.
    """
    valor = p*0.5+r*0.3+a*0.2
    return round(valor*(1+n*0.1),3)

class MRLanguageCountriesBudgets(MRJob):
    def mapper(self, _, linea):
        # Procesamos el archivo
        temporada, jugador, pts, reb, ast, net = procesaArchivo(linea)
        # Calculamos la valoración de cada jugador 
        eff = calculaValor(pts, reb, ast, net)
        
        # Generamos los pares clave (temporada) y valor (valoración
        # y nombre del jugador)
        yield temporada, (eff, jugador)
    
    def reducer(self, temporada, values):
        # Hallamos el máximo de valoración por temporada
        mvp = max(values)
        yield temporada, mvp

# Para permitir la ejecucion por linea de comandos
if __name__ == '__main__':
    MRLanguageCountriesBudgets.run()