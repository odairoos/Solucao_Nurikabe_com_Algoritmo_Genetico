"""
@author: Odair
"""

# -*- coding: utf-8 -*-
# Instala a biblioteca para implementação do algoritmo genético
#!pip install geneticalgorithm
# Ver documentação: pypi.org/project/geneticalgorithm/

"""
Created on Sun Jun 16 15:51:45 2024

@author: Odair
"""
import numpy as np
import matplotlib.pyplot as plt
from geneticalgorithm2 import geneticalgorithm2 as ga
from game_logic import NurikabeGame
from game_render import print_board, ask_for_move

import numpy as np

class Genetics_Game:
    def __init__(self, game, size, generations=20000):
        self.game = game
        self.size = size
        self.puzzle = np.array(game.board)  # Certificando que puzzle é um array NumPy
        self.generations = generations

    def fill_island(self, individual, x, y, count):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < individual.shape[0] and 0 <= ny < individual.shape[1] and count > 0:
                if individual[nx, ny] == 0:
                    individual[nx, ny] = -1
                    count -= 1
                elif individual[nx, ny] == -1:
                    count -= 1

    def generate_initial_population(self):
        population = []
        for _ in range(self.size):
            individual = np.copy(self.puzzle)
            for x in range(individual.shape[0]):
                for y in range(individual.shape[1]):
                    if individual[x, y] > 0:  # Se a célula contém um valor positivo
                        self.fill_island(individual, x, y, individual[x, y])

            # Preenchendo os espaços vazios restantes
            for x in range(individual.shape[0]):
                for y in range(individual.shape[1]):
                    if individual[x, y] == 0:
                        individual[x, y] = np.random.choice([-2, -1])

            population.append(individual)
        return population
    
    # -- agora começam as regras do Nurikabe
    def verifica_water_connected(self, board):
        visited = np.zeros((self.size, self.size), dtype=bool)
        water_cells = np.argwhere(board == -2)

        if not len(water_cells):
            return True  # Se não há água, consideramos como conectada.

        def dfs(x, y):
            if x < 0 or x >= self.size or y < 0 or y >= self.size or visited[x, y] or board[x, y] != -2:
                return
            visited[x, y] = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                dfs(x + dx, y + dy)

        dfs(water_cells[0][0], water_cells[0][1])

        return np.all(visited[water_cells[:, 0], water_cells[:, 1]])
    
    ## regra original alterada para estrutura NUMPY
    def tamanho_island_sizes(self, board):
        visited = np.zeros((self.size, self.size), dtype=bool)

        def dfs(x, y, target):
            if x < 0 or x >= self.size or y < 0 or y >= self.size or visited[x, y] or (board[x, y] != -1 and board[x, y] != target):
                return 0
            visited[x, y] = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            count = 1
            for dx, dy in directions:
                count += dfs(x + dx, y + dy, target)
            return count

        for x in range(self.size):
            for y in range(self.size):
                if board[x, y] >= 1 and not visited[x, y]:
                    island_size = board[x, y]
                    counted_size = dfs(x, y, island_size)
                    if counted_size != island_size:
                        return False
        return True

    def islands_separated_correctly(self, board):
        visited = np.zeros((self.size, self.size), dtype=bool)

        def dfs(x, y, island_value):
            if x < 0 or x >= self.size or y < 0 or y >= self.size or visited[x, y] or board[x, y] == -2:
                return False
            if board[x, y] >= 1 and board[x, y] != island_value:
                return True
            visited[x, y] = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            return any(dfs(x + dx, y + dy, island_value) for dx, dy in directions)

        for x in range(self.size):
            for y in range(self.size):
                if board[x, y] >= 1 and not visited[x, y]:
                    if dfs(x, y, board[x, y]):
                        return False
        return True

    def calcula_islands_and_water(self, board):
        total_cells = self.size * self.size
        num_islands = 0
        num_water = 0

        for row in board:
            for cell in row:
                if cell >= 1:
                    num_islands += cell
                elif cell == -2:
                    num_water += 1

        return total_cells - num_islands == num_water

    def no_vazio_spaces(self, board):
        for row in board:
            if 0 in row:
                return False
        return True
    ## --- aqui terminam as regras do Nurikabe 
    
    ## --- A função de FITNESS aplica recompensas por cada regra do Nurikabe válida
    ##     Foram retiradas as penalidades pois não ajudavam na convergencia para a solução
    ##
    def fitness_function(self, solution):
        size = int(len(solution) ** 0.5)
        board = np.array(solution).reshape((size, size))

        fitness = 0
        if self.verifica_water_connected(board):
            fitness += 1
        if self.tamanho_island_sizes(board):
            fitness += 2
        if self.islands_separated_correctly(board):
            fitness += 3
        if self.calcula_islands_and_water(board):
            fitness += 4
        if self.no_vazio_spaces(board):
            fitness += 5
        return fitness

    def evaluate(self, individual):
        return self.fitness_function(individual.flatten())

    def select_parents(self, population, scores):
        idx = np.argsort(scores)
        return [population[i] for i in idx[-2:]]

    def crossover(self, parent1, parent2):
        point = np.random.randint(1, parent1.size - 1)
        child1 = np.concatenate((parent1.flatten()[:point], parent2.flatten()[point:])).reshape(parent1.shape)
        child2 = np.concatenate((parent2.flatten()[:point], parent1.flatten()[point:])).reshape(parent1.shape)
        return child1, child2

    def mutate(self, individual, mutation_rate=0.01):
        for x in range(individual.shape[0]):
            for y in range(individual.shape[1]):
                if np.random.rand() < mutation_rate:
                    individual[x, y] = np.random.choice([-2, -1])
        return individual

    def genetico(self):
        population = self.generate_initial_population()
        for generation in range(self.generations):
            scores = [self.evaluate(ind) for ind in population]
            if max(scores) == np.prod(self.puzzle.shape):
                break
            parents = self.select_parents(population, scores)
            next_population = []
            while len(next_population) < len(population):
                parent1, parent2 = np.random.choice(len(parents), 2, replace=False)
                child1, child2 = self.crossover(parents[parent1], parents[parent2])
                next_population.extend([self.mutate(child1), self.mutate(child2)])
            population = next_population[:len(population)]

        best_individual = population[np.argmax(scores)]
        return best_individual
