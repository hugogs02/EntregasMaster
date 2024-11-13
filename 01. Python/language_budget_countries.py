import sys
from mrjob.job import MRJob
from collections import defaultdict

class MRLanguageCountriesBudgets(MRJob):
    def mapper(self, _, linea):
        # Para cada línea, obtenemos sus componentes, que están separados por |
        titulo, anho, idioma, pais, presupuesto = linea.split('|')
        
        # Intentamos convertir a int el año y el presupuesto
        try:
            anho = int(anho)
            presupuesto = int(presupuesto)
        except:
            pass
        
        # Si se conocen los datos de la película (es decir, existen el idioma y
        # país y el presupuesto y año existen o son distintos de -1)
        if idioma and pais and presupuesto and anho \
            and presupuesto != -1 and anho != -1:
            # Generamos los pares de clave-valor, con idioma como clave, y
            # el país y presupuesto como valor
            yield idioma, (pais, int(presupuesto))
    
    def reducer(self, idioma, values):
        # Generamos un defaultdict para el presupuesto 
        pais_presupuesto = defaultdict(int)
        # Iteramos sobre los valores recibidos del mapper
        for pais, presupuesto in values:
            # Sumamos el presupuesto de cada país
            pais_presupuesto[pais] += presupuesto
        # Producimos los pares clave-valor, donde la clave es el idioma, y
        # los valores una lista con cada país y la suma de los presupuestos
        yield idioma, (list(pais_presupuesto.keys()),
                       sum(pais_presupuesto.values()))

# Para permitir la ejecucion por linea de comandos
if __name__ == '__main__':
    MRLanguageCountriesBudgets.run()