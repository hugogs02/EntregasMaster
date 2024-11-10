import sys
from mrjob.job import MRJob
from collections import defaultdict

def procesaArchivo(line):
    linea = line.split(',')
    temporada = linea[1]
    jugador = linea[0]
    pts = float(linea[3])
    reb = float(linea[4])
    ast = float(linea[5])
    net = float(linea[6])

    return temporada, jugador, pts, reb, ast, net

def calculaValor(p, r, a, n):
    valor = p*0.5+r*0.3+a*0.2
    return round(valor*(1+n*0.1),3)

class MRLanguageCountriesBudgets(MRJob):
    def mapper(self, _, linea):
        temporada, jugador, pts, reb, ast, net = procesaArchivo(linea)
        eff = calculaValor(pts, reb, ast, net)
        
        yield temporada, (eff, jugador)
    
    def reducer(self, temporada, values):
        mvp = max(values)
        yield temporada, mvp

# Para permitir la ejecucion por linea de comandos
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("El programa debe tomar 3 argumentos:")
        print("language_budget_countries.py -q archivo_entrada")
        sys.exit()
    
    archivo = sys.argv[2]
    MRLanguageCountriesBudgets.run()