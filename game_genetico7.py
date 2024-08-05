# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:02:05 2024

@author: Odair
"""

import numpy as np

import matplotlib.pyplot as plt

class Genetics_Game:
    def __init__(self, game, size, generations=70, population_size=6500):
        self.game = game
        self.size = size
        self.puzzle = np.array(game.board)  # Certificando que puzzle é um array NumPy
        self.generations = generations
        self.population_size = population_size

    def plot_convergence(self, best_scores, mean_scores, std_scores):
        generations = range(len(best_scores))
        
               
        plt.figure(figsize=(14, 7))
        plt.plot(generations, best_scores, label='Melhor Pontuação')
        plt.plot(generations, mean_scores, label='Pontuação Média')
        plt.fill_between(generations, 
                         np.array(mean_scores) - np.array(std_scores), 
                         np.array(mean_scores) + np.array(std_scores), 
                         color='b', alpha=0.2, label='Desvio Padrão')
        
        plt.xlabel('Gerações')
        plt.ylabel('Pontuação')
        plt.title('Convergência do Algoritmo Genético')
        plt.legend()
        plt.show()
        
    def fill_island(self, individual, x, y, count):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        stack = [(x, y)]
        filled = 1  # Inclui a célula numerada
        original_value = individual[x, y]

        # Marca a célula numerada como parte da ilha
        if individual[x, y] != -1:
            individual[x, y] = -1

        while stack and filled < count:
            cx, cy = stack.pop()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < individual.shape[0] and 0 <= ny < individual.shape[1]:
                    if individual[nx, ny] == 0 and filled < count:
                        individual[nx, ny] = -1
                        filled += 1
                        stack.append((nx, ny))
                    elif individual[nx, ny] == -1:
                        continue

        # Verificar se a ilha foi preenchida completamente
        if filled < count:
            print("Atenção: Não foi possível preencher a ilha completamente.")

        # Preencher o restante das células adjacentes com -2
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < individual.shape[0] and 0 <= ny < individual.shape[1]:
                    if individual[nx, ny] == 0:
                        individual[nx, ny] = -2

        # Restaurar o valor original da célula numerada
        individual[x, y] = original_value

    def generate_initial_population(self):
        population = []
        for _ in range(self.population_size):
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
    
    def no_2x2_water_blocks(self, board):
        for x in range(self.size - 1):
            for y in range(self.size - 1):
                if board[x, y] == -2 and board[x + 1, y] == -2 and board[x, y + 1] == -2 and board[x + 1, y + 1] == -2:
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
        if self.no_2x2_water_blocks(board):
            fitness += 6  # Adicionando um valor ao fitness se não houver blocos 2x2 de água
        return fitness

    def evaluate(self, individual):
        return self.fitness_function(individual.flatten())

    def select_parents(self, population, scores):
        idx = np.argsort(scores)
        return [population[i] for i in idx[-2:]]   # particiona pegando 2 individuos conforme score

    def crossover(self, parent1, parent2):
        # Criar cópias dos pais para preservar os originais
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
    
        # Converter os pais em arrays 1D para facilitar o cruzamento
        flat_parent1 = parent1.flatten()
        flat_parent2 = parent2.flatten()
    
        # Inicializar os filhos como cópias dos pais
        flat_child1 = flat_parent1.copy()
        flat_child2 = flat_parent2.copy()
        # Realizar o cruzamento somente nas células com -1 ou -2
        for i in range(len(flat_parent1)):
            if flat_parent1[i] in [-1, -2] and flat_parent2[i] in [-1, -2]:
                if np.random.rand() < 0.5:  # Adicionar um fator de aleatoriedade para o cruzamento
                    flat_child1 = flat_parent1.copy()
                    flat_child2 = flat_parent2.copy()
                    flat_child1[i], flat_child2[i] = flat_child2[i], flat_child1[i]
    
        # Remodelar os filhos de volta para a forma original
        child1 = flat_child1.reshape(parent1.shape)
        child2 = flat_child2.reshape(parent1.shape)
    
        return child1, child2


    def mutate(self, individual, mutation_rate=0.01):
        for x in range(individual.shape[0]):
            for y in range(individual.shape[1]):
                if individual[x, y] in [-1, -2] and np.random.rand() < mutation_rate:
                    individual[x, y] = np.random.choice([-2, -1])
        return individual
    
             
    def genetico(self):
        population = self.generate_initial_population()
        best_scores = []
        mean_scores = []
        std_scores = []
        for generation in range(self.generations):
            scores = [self.evaluate(ind) for ind in population]
            best_scores.append(max(scores))
            mean_scores.append(np.mean(scores))
            std_scores.append(np.std(scores))
            
            if max(scores) == np.prod(self.puzzle.shape):
                break
            parents = self.select_parents(population, scores)
            next_population = []
            while len(next_population) < len(population):
                parent1, parent2 = np.random.choice(len(parents), 2, replace=False)
                child1, child2 = self.crossover(parents[parent1], parents[parent2])
                next_population.extend([self.mutate(child1), self.mutate(child2)])
            population = next_population[:len(population)]
            #print(population,"#")
        best_individual = population[np.argmax(scores)]
        self.plot_convergence(best_scores, mean_scores, std_scores)
        return best_individual, scores
    
    
    
         
# Exemplo de uso:
# game é um objeto que contém a matriz do tabuleiro inicial do Nurikabe
# size é o tamanho do tabuleiro (por exemplo, 5 para um tabuleiro 5x5)

# genetics_game = Genetics_Game(game, size)
# best_solution, final_scores, best_scores_per_gen = genetics_game.genetico()

# print("Melhor solução encontrada:")
# print(best_solution)
# print("Pontuação final da melhor solução:", max(final_scores))
# print("Pontuação ao longo das gerações:", best_scores_per_gen)
