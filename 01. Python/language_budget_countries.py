import sys
from mrjob.job import MRJob
from collections import defaultdict

class MRLanguageCountriesBudgets(MRJob):
    def mapper(self, _, linea):
        titulo, anho, idioma, pais, presupuesto = linea.split('|')
        try:
            anho = int(anho)
            presupuesto = int(presupuesto)
        except:
            pass
        
        if idioma and pais and presupuesto and anho \
            and presupuesto != -1 and anho != -1:
            yield idioma, (pais, int(presupuesto))
    
    def reducer(self, idioma, values):
        pais_presupuesto = defaultdict(int)
        for pais, presupuesto in values:
            pais_presupuesto[pais] += presupuesto
        yield idioma, (list(pais_presupuesto.keys()),
                       sum(pais_presupuesto.values()))

# Para permitir la ejecucion por linea de comandos
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("El programa debe tomar 3 argumentos:")
        print("language_budget_countries.py -q archivo_entrada")
        sys.exit()
    
    archivo = sys.argv[2]
    MRLanguageCountriesBudgets.run()