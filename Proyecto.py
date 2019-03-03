# Programación de algoritmos para determinar la existencia de circuitos
# y rutas de Euler y, circuitos de Hamilton

# Para la clase
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate
from functools import reduce
import itertools

# Para el uso de la clase
import random

# En esta clase las aristas son listas de pares ordenados, no se admiten grafos con
# lados múltiples
class Grafo():
    # Método constructor
    def __init__(self, aristas = None, dirigido = False, *args):
        self.dirigido = dirigido
        self.aristas_pesos = []

        if aristas == []:
            aristas = None

        if aristas != None:
            if self.dirigido == False:
                self.aristas = list(map(tuple,
                                   self.DeleteDuplicates(list(map(set, aristas)))))
                naristas = []
                for i in self.aristas:
                    if (len(i) != 1):
                        naristas = naristas + [i]
                    else:
                        naristas = naristas + [(i[0], i[0])]
                self.aristas = naristas
            else:
                self.aristas = self.DeleteDuplicates(aristas)

            self.vertices = sorted(set(list(reduce(lambda set1, set2: set1 | set2,
                                                   list(map(set, self.aristas))))), key=str)

            self.nuevas_aristas = self.aristas
        else:
            self.aristas = []
            self.vertices = []
            self.nuevas_aristas = []

    # Método para eliminar duplicados
    def DeleteDuplicates(self, lista, *args):
        def eliminateComp(olista, m):
            olista = [x if x != m else None for x in olista]
            olista[olista.index(None)] = m
            while(olista.count(None) != 0):
                olista.pop(olista.index(None))
            return olista

        list = lista
        for i in range(len(lista)):
            list = eliminateComp(list, lista[i])
        return list

    # Método para actualizar la lista de vértices del grafo
    def ListaVertices(self, aristas, *args):
        self.vertices = sorted(set(list(set(self.vertices) |
                                 set(list(reduce(lambda set1, set2: set1 | set2,
                                   list(map(set, aristas))))))), key = str)

    # Método para añadir aristas al grafo
    def AddAristas(self, aristas, *args):
        if self.dirigido == False:
            aristas = list(map(tuple,
                                self.DeleteDuplicates(list(map(set, self.aristas + aristas)))))

            naristas = []
            for i in aristas:
                if (len(i) != 1):
                    naristas = naristas + [i]
                else:
                    naristas = naristas + [(i[0], i[0])]
            naristas = self.DeleteDuplicates(naristas)
        else:
            naristas = self.DeleteDuplicates(self.aristas + aristas)

        self.nuevas_aristas = naristas[len(self.aristas):len(naristas)]
        self.aristas = naristas
        self.ListaVertices(naristas)

    # Método para agregar pesos a las aristas del grafo
    def AddPesos(self, list_pesos, *args):
        def Agregar_peso(tupla, peso):
            new_tupla = list(tupla)
            new_tupla.append(peso)
            return tuple(new_tupla)

        if len(self.nuevas_aristas) == len(list_pesos):
            self.aristas_pesos = self.aristas_pesos + [Agregar_peso(self.nuevas_aristas[i], list_pesos[i])
                                                       for i in list(range(len(list_pesos)))]
        if len(self.aristas) == len(list_pesos):
            self.aristas_pesos = [Agregar_peso(self.aristas[i], list_pesos[i])
                                                       for i in list(range(len(list_pesos)))]

    # Método para construir la matriz de adyacencia del grafo
    def MatrixAdyacencia(self, tabla = False, *args):
        Matriz = []

        if self.vertices != []:
            if self.dirigido == False:
                Matriz = np.zeros((len(self.vertices),len(self.vertices)))

                for i in range(self.NumVertex()):
                    for j in range(self.NumVertex()):
                        if self.aristas.count((self.vertices[i],self.vertices[j])) != 0:
                            if(i == j):
                                Matriz[i][j] = 2
                            else:
                                Matriz[i][j] = 1
                        else:
                            if self.aristas.count((self.vertices[j], self.vertices[i])) != 0:
                                Matriz[i][j] = 1
            else:
                Matriz = np.zeros((len(self.vertices), len(self.vertices)))

                for i in range(self.NumVertex()):
                    for j in range(self.NumVertex()):
                        if self.aristas.count((self.vertices[i], self.vertices[j])) != 0:
                            Matriz[i][j] = 1
        else:
            return "Grafo vacío"

        if self.vertices != []:
            if not tabla:
                return Matriz
            else:
                return tabulate(Matriz,
                               headers = self.vertices,
                               tablefmt = "fancy_grid",
                               stralign = "center",
                               showindex = self.vertices)

    # Método para determinar el número de vértices
    def NumVertex (self, *args):
        return len(self.vertices)

    # Método para determinar el número de lados
    def NumEdges (self, *args):
        return len(self.aristas)

    # Método para construir la matriz de adyacencia de pesos del grafo
    def MatrixAdyacenciaPesos(self, tabla = False, *args):
        Matriz = []

        if self.vertices != []:
            if self.dirigido == False:
                grafo_auxiliar = nx.Graph()
                grafo_auxiliar.add_weighted_edges_from(self.aristas_pesos)
                Matriz = np.asarray(nx.to_numpy_matrix(grafo_auxiliar,
                                                        nodelist = self.vertices))
            else:
                grafo_auxiliar = nx.DiGraph()
                grafo_auxiliar.add_weighted_edges_from(self.aristas_pesos)
                Matriz = np.asarray(nx.to_numpy_matrix(grafo_auxiliar,
                                                        nodelist = self.vertices))
        else:
            return "Grafo vacío"

        if self.vertices != []:
            if not tabla:
                return Matriz
            else:
                return tabulate(Matriz,
                               headers = self.vertices,
                               tablefmt = "fancy_grid",
                               stralign = "center",
                               showindex = self.vertices)

    # Método para visualizar el grafo (no muestra lazos)
    def Dibujar(self, archivo = False, *args):
        if self.dirigido == False:
            grafo_auxiliar = nx.Graph()
            grafo_auxiliar.add_edges_from(self.aristas)
            nx.draw(grafo_auxiliar, node_size=500,
                    node_color='g',
                    with_labels=True, font_weight='bold')
            if archivo == False:
                plt.show()
            else:
                plt.savefig("Grafo.png")
        else:
            grafo_auxiliar = nx.DiGraph()
            grafo_auxiliar.add_edges_from(self.aristas)
            nx.draw(grafo_auxiliar, node_size=500,
                    node_color='g',
                    with_labels=True, font_weight='bold')
            if archivo == False:
                plt.show()
            else:
                plt.savefig("Grafo.png")

    # Método para obtener las valencias de los nodos
    def Grados(self, tabla = False, *args):
        if self.vertices != []:
            valencias = []
            valencias_internas = []
            valencias_externas = []

            if self.dirigido == False:
                Matriz_aux = self.MatrixAdyacencia()
                valencias = [sum(Matriz_aux[i]) for i in range(self.NumVertex())]
            else:
                Matriz_aux = np.array(self.MatrixAdyacencia()).transpose()
                valencias_internas = [sum(Matriz_aux[i]) for i in range(self.NumVertex())]
                Matriz_aux = self.MatrixAdyacencia()
                valencias_externas = [sum(Matriz_aux[i]) for i in range(self.NumVertex())]

            if not tabla:
                if self.dirigido == False:
                    return(valencias)
                else:
                    return([valencias_internas, valencias_externas])
            else:
                if self.dirigido == False:
                    return tabulate([valencias],
                                   headers=self.vertices,
                                   tablefmt="fancy_grid",
                                   stralign="center",
                                   showindex=["Grados o valencias"])
                else:
                    return tabulate([valencias_internas, valencias_externas],
                                   headers=self.vertices,
                                   tablefmt="fancy_grid",
                                   stralign="center",
                                   showindex=["Grados o valencias internas",
                                              "Grados o valencias externas"])
        else:
            return "Grafo vacío"

    # Método para construir las aristas de un grafo mediante
    # su matriz de adyacencia convencional o de pesos
    def Grafo_matrix(self, matrix, lista_vertices, directed = False, *args):
        def SimetricaQ(omatrix):
            EsSimetrica = True

            matrix_auxiliar1 = np.array(omatrix)
            matrix_auxiliar2 = matrix_auxiliar1.transpose()
            dimension = matrix_auxiliar1.shape

            for i in range(dimension[0]):
                for j in range(dimension[1]):
                    if matrix_auxiliar1[i][j] != matrix_auxiliar2[i][j]:
                        EsSimetrica = False

            if EsSimetrica:
                return True
            else:
                return False

        matrix_auxiliar = np.array(matrix)
        dimensions = matrix_auxiliar.shape

        if dimensions[0] == dimensions[1] & len(lista_vertices) == dimensions[0]:
            if not directed:
                if SimetricaQ(matrix_auxiliar):
                    self.dirigido = directed
                    self.vertices = lista_vertices
                    self.aristas = []

                    for i in range(dimensions[0]):
                        for j in range(dimensions[1]):
                            if matrix_auxiliar[i][j] != 0:
                                if self.aristas.count((lista_vertices[j],
                                                     lista_vertices[i])) == 0:
                                    self.aristas = self.aristas + \
                                                   [(lista_vertices[i],
                                                     lista_vertices[j])]
                else:
                    return "La matriz ingresada no es simétrica"
            else:
                self.dirigido = directed
                self.vertices = lista_vertices
                self.aristas = []

                for i in range(dimensions[0]):
                    for j in range(dimensions[1]):
                        if matrix_auxiliar[i][j] != 0:
                            self.aristas = self.aristas + \
                                           [(lista_vertices[i],
                                             lista_vertices[j])]
        else:
            return "No es una matriz cuadrada o la cantidad de vértices es inválida"

    # Método para determinar la existencia de un circuito de Euler
    def CircuitoEulerianoQ(self, *args):
        if self.dirigido == False:
            if self.vertices != []:
                valencias_impares = [i for i in self.Grados() if i % 2 != 0]
                if valencias_impares != []:
                    return False
                else:
                    grafo_auxiliar = nx.Graph()
                    grafo_auxiliar.add_edges_from(self.aristas)
                    return "({}, {})".format(True, list(nx.eulerian_circuit(grafo_auxiliar)))
            else:
                return False
        else:
            return "NaD"

    # Método para determinar la existencia de una ruta de Euler
    def RutaEulerianaQ(self, *args):
        if self.dirigido == False:
            if self.vertices != []:
                valencias_impares = [i for i in self.Grados() if i % 2 != 0]
                if len(valencias_impares) == 2:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return "NaD"

    # Método de ORE (existencia de un circuito hamiltoniano)
    def ORE(self):
        if self.dirigido == False:
            EsORE = True
            n = self.NumVertex()
            Matriz = self.MatrixAdyacencia()

            if [Matriz[i][i] for i in range(n)].count(0) == n & n >= 3:
                producto_cartesiano = list(itertools.product(self.vertices, self.vertices))
                No_adyacentes = [i for i in list(set(producto_cartesiano) - set(self.aristas))
                                 if i[0] != i[1]]

                for i in No_adyacentes:
                    Valencias = self.Grados()
                    Pos_x = self.vertices.index(i[0])
                    Pos_y = self.vertices.index(i[1])
                    if Valencias[Pos_x] + Valencias[Pos_y] < n:
                        EsORE = False
                        return "No se cumple la propiedad de ORE, un contraejemplo " \
                               "ocurre en los nodos {} y {}".format(self.vertices[Pos_x],self.vertices[Pos_y])
                if EsORE:
                    return "Se cumple la propiedad de ORE, existe un circuito de Hamilton"
            else:
                return "El grafo no es simple, o bien, la cantidad de nodos " \
                       "no es mayor o igual a tres"
        else:
            return "NaD"

    # Método de Dirac (existencia de un circuito hamiltoniano)
    def DIRAC(self):
        if self.dirigido == False:
            EsDIRAC = True
            n = self.NumVertex()
            Matriz = self.MatrixAdyacencia()

            if [Matriz[i][i] for i in range(n)].count(0) == n & n >= 3:
                Valencias = self.Grados()

                for i in Valencias:
                    if i < n/2:
                        EsDIRAC = False
                        return "No se cumple la propiedad de DIRAC, un contraejemplo " \
                               "ocurre en el vértice {}"\
                            .format(self.vertices[Valencias.index(i)])
                if EsDIRAC:
                    return "Se cumple la propiedad de DIRAC, existe un circuito de Hamilton"
            else:
                return "El grafo no es simple, o bien, la cantidad de nodos " \
                       "no es mayor o igual a tres"
        else:
            return "NaD"

    # Método de las aristas (existencia de un circuito hamiltoniano)
    def NumAristas(self):
        if self.dirigido == False:
            EsNumAristas = True
            n = self.NumVertex()
            Matriz = self.MatrixAdyacencia()

            if [Matriz[i][i] for i in range(n)].count(0) == n & n >= 3:
                if self.NumEdges() < (n**2-3*n+6)/2:
                    EsNumAristas = False
                    return "No se cumple la propiedad del número de lados, " \
                           "hay {} aristas y la fórmula da {}"\
                        .format(self.NumEdges(),(n**2-3*n+6)/2)
                if EsNumAristas:
                    return "Se cumple la propiedad del número de lados, " \
                           "existe un circuito de Hamilton"
            else:
                return "El grafo no es simple, o bien, la cantidad de nodos " \
                       "no es mayor o igual a tres"
        else:
            return "NaD"

# ------------------------- Usando la clase -------------------------
# -------------------------------------------------------------------

# ------------ Creando grafos con un conjunto de aristas ------------
# -------------------------------------------------------------------

aristas = [("a", "b"), (1, 1), (1, 2), ("b", "a"), (2, 3), (3, 2), (3, 3),
           (6, 4), (8, 2), (4, 4), (10, 20), (6, 4)]
grafo1 = Grafo(aristas) # Grafo no dirigido
grafo2 = Grafo(aristas, True) # Grafo dirigido

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# ------- Imprimiendo los vértices y los lados de los grafos --------
# -------------------------------------------------------------------

# Grafo no dirigido
print("Vértices del grafo no dirigido: {}".format(grafo1.vertices))
print("Aristas del grafo no dirigido: {}".format(grafo1.aristas))

# -------------------------------------------------------------------

# Grafo dirigido
print("Vértices del grafo dirigido: {}".format(grafo2.vertices))
print("Aristas del grafo dirigido: {}".format(grafo2.aristas))

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# -- Imprimiento la cantidad de vértices y de lados de los grafos ---
# -------------------------------------------------------------------

# Grafo no dirigido
print("El grafo no dirigido tiene {} vértices y {} lados"
      .format(grafo1.NumVertex(), grafo1.NumEdges()))

# -------------------------------------------------------------------

# Grafo dirigido
print("El grafo dirigido tiene {} vértices y {} lados"
      .format(grafo2.NumVertex(), grafo2.NumEdges()))

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# ----------------- Añadiendo aristas a los grafos ------------------
# -------------------------------------------------------------------

# Grafo no dirigido
grafo1.AddAristas([("d", "e"), ("e", "f"), (7, 4), (7, 7), (7, 2)])
print("Vértices actualizados del grafo no dirigido: {}".format(grafo1.vertices))
print("Aristas actualizadas del grafo no dirigido: {}".format(grafo1.aristas))
print(grafo1.NumEdges())

# -------------------------------------------------------------------

# Grafo dirigido
grafo2.AddAristas([("d", "e"), ("e", "f"), (7, 4), (7, 7), (7, 2)])
print("Vértices actualizados del grafo dirigido: {}".format(grafo2.vertices))
print("Aristas actualizadas del grafo dirigido: {}".format(grafo2.aristas))
print(grafo2.NumEdges())

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# ----------- Añadiendo pesos a las aristas de los grafos -----------
# -------------------------------------------------------------------

# Grafo no dirigido
pesos = [random.uniform(1, 15) for i in range(grafo1.NumEdges())]
grafo1.AddPesos(pesos)
print("Lista de aristas-pesos del grafo no dirigido: {}".format(grafo1.aristas_pesos))

# -------------------------------------------------------------------

# Grafo dirigido
pesos = [random.uniform(1, 15) for j in range(grafo2.NumEdges())]
grafo2.AddPesos(pesos)
print("Lista de aristas-pesos del grafo dirigido: {}".format(grafo2.aristas_pesos))

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# ----------------- Matriz y/o tabla de adyacencia -----------------
# -------------------------------------------------------------------

# Grafo no dirigido
print("Matriz de adyacencia del grafo no dirigido:")
print(grafo1.MatrixAdyacencia())
print("Tabla de adyacencia del grafo no dirigido:")
print(grafo1.MatrixAdyacencia(True))

# -------------------------------------------------------------------

# Grafo dirigido
print("Matriz de adyacencia del grafo dirigido:")
print(grafo2.MatrixAdyacencia())
print("Tabla de adyacencia del grafo dirigido:")
print(grafo2.MatrixAdyacencia(True))

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# ------------- Matriz y/o tabla de adyacencia de pesos -------------
# -------------------------------------------------------------------

# Grafo no dirigido
print("Matriz de adyacencia de pesos del grafo no dirigido:")
print(grafo1.MatrixAdyacenciaPesos())
print("Tabla de adyacencia de pesos del grafo no dirigido:")
print(grafo1.MatrixAdyacenciaPesos(True))

# -------------------------------------------------------------------

# Grafo dirigido
print("Matriz de adyacencia de pesos del grafo dirigido:")
print(grafo2.MatrixAdyacenciaPesos())
print("Tabla de adyacencia de pesos del grafo dirigido:")
print(grafo2.MatrixAdyacenciaPesos(True))

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# ---------------------- Dibujando los grafos -----------------------
# -------------------------------------------------------------------

# Grafo no dirigido
grafo1.Dibujar()
grafo1.Dibujar(True) # Genera un archivo .png

# -------------------------------------------------------------------

# Grafo dirigido
grafo2.Dibujar()
grafo2.Dibujar(True) # Genera un archivo .png

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# ------------- Valencias de los vértices de los grafos -------------
# -------------------------------------------------------------------

# Grafo no dirigido
print(grafo1.Grados())
print(grafo1.Grados(True)) # Genera una tabla de valencias

# -------------------------------------------------------------------

# Grafo dirigido
print(grafo2.Grados())
print(grafo2.Grados(True)) # Genera una tabla de valencias

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# -------- Creando un grafo con una matriz de adyacencia dada -------
# -------------------------------------------------------------------

# Grafo no dirigido
# -------------------------------------------------------------------
# Método que genera una matriz cuadrada seudoaleatoria con valores
# de 1 a m
# -------------------------------------------------------------------
def matriz_pseudoaleatoria(n, m):
    matriz_auxiliar = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matriz_auxiliar[i][j] = random.randrange(0, m)

    return matriz_auxiliar
# -------------------------------------------------------------------
# Método que transforma una matriz en simétrica
# -------------------------------------------------------------------
def simetrica_pseudoaleatoria(Mtr, Con_lazos = True):
    matrix_auxiliar = np.array(Mtr)
    dimensions = matrix_auxiliar.shape

    if Con_lazos:
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                if i > j:
                    matrix_auxiliar[i][j] = matrix_auxiliar[j][i]
    else:
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                if i > j:
                    matrix_auxiliar[i][j] = matrix_auxiliar[j][i]
                if i == j:
                    matrix_auxiliar[i][j] = 0

    return matrix_auxiliar
# -------------------------------------------------------------------

grafo3 = Grafo()
grafo3.Grafo_matrix(simetrica_pseudoaleatoria(matriz_pseudoaleatoria(5, 10)),
                    [1, 2, 3, 4, 5], False)
print("Vértices del grafo no dirigido: {}".format(grafo3.vertices))
print("Aristas del grafo no dirigido: {}".format(grafo3.aristas))

# -------------------------------------------------------------------

# Grafo dirigido
grafo4 = Grafo()
grafo4.Grafo_matrix(matriz_pseudoaleatoria(5, 10),
                    [1, 2, 3, 4, 5], True)
print("Vértices del grafo dirigido: {}".format(grafo4.vertices))
print("Aristas del grafo dirigido: {}".format(grafo4.aristas))

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# --------------- Existencia de un circuito de Euler ----------------
# -------------------------------------------------------------------

print("El grafo 1 ¿tiene un circuito euleriano?: {}"
      .format(grafo1.CircuitoEulerianoQ()))
print("El grafo 2 ¿tiene un circuito euleriano?: {}"
      .format(grafo2.CircuitoEulerianoQ()))
print("El grafo 3 ¿tiene un circuito euleriano?: {}"
      .format(grafo3.CircuitoEulerianoQ()))
print("Observe sus valencias: ")
print(grafo3.Grados(True))

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# ----------------- Existencia de una ruta de Euler -----------------
# -------------------------------------------------------------------

print("El grafo 1 ¿tiene una ruta euleriana?: {}"
      .format(grafo1.RutaEulerianaQ()))
print("El grafo 2 ¿tiene una ruta euleriana?: {}"
      .format(grafo2.RutaEulerianaQ()))
print("El grafo 3 ¿tiene una ruta euleriana?: {}"
      .format(grafo3.RutaEulerianaQ()))
print("Observe sus valencias: ")
print(grafo3.Grados(True))

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# ---------------------- ORE - DIRAC - Aristas ----------------------
# -------------------------------------------------------------------

grafo5 = Grafo()
grafo5.Grafo_matrix(simetrica_pseudoaleatoria(matriz_pseudoaleatoria(5, 5),
                                              False), [1, 2, 3, 4, 5], False)
print("Valencias del grafo pseudoaleatorio:")
print(grafo5.Grados(True))
print(grafo5.ORE())
print(grafo5.DIRAC())
print(grafo5.NumAristas())

# -------------------------------------------------------------------
# -------------------------------------------------------------------




