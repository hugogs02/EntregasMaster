import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import FuncionesMineria

datos = pd.read_excel("DatosElecciones.xlsx", sheet_name='DatosEleccionesEspaña')
print(datos.dtypes)