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
from geneticalgorithm import geneticalgorithm as ga
from game_logic import NurikabeGame
from game_render import print_board, ask_for_move

class Genetics_Game:
    # Construtor
    def __init__(self, game, size):
        self.size = size  # Define o atributo 'size'
        if isinstance(game, NurikabeGame):
            self.puzzle = game.board  # Supondo que o jogo tenha um atributo board
            self.board = np.array(game.board)
        else:
            self.puzzle = game
            self.board = np.array(game)
           
           
    def get_puzzle(self):
        return self.puzzle

    def corrigir(self, matriz):
        # Encontrar posições onde os valores são iguais a zero
        valor_zero = np.argwhere(matriz == 0)
        # Gerar valores aleatórios entre -1 e 1 nas posições encontradas
        alternador = -1
        # Substituir os valores não positivos pelos valores aleatórios
        for indice in valor_zero:
            matriz[tuple(indice)] = alternador
            alternador *= -1
        return matriz 

    def check_block_pools(self):
        """
        Verifica a existência de blocos 2x2 de mar no tabuleiro.
        :return: True se uma "piscina" 2x2 for encontrada, False caso contrário.
        """
        for x in range(self.size - 1):
            for y in range(self.size -1):
                if self.board[x][y] == -2 and \
                   self.board[x+1][y] == -2 and \
                   self.board[x][y+1] == -2 and \
                   self.board[x+1][y+1] == -2:
                    return True
        return False
    
    def verifica_water_connected(self):
        """
        Verifica se toda a água no tabuleiro está conectada.
        :return: True se toda a água estiver conectada, False caso contrário.
        """
        visited = np.zeros((self.size, self.size), dtype=bool)
        water_cells = np.argwhere(self.board == -2)

        if not len(water_cells):
            return True  # Se não há água, consideramos como conectada.

        # Função auxiliar para realizar a DFS a partir de uma célula de água
        def dfs(x, y):
            if x < 0 or x >= self.size or y < 0 or y >= self.size or visited[x, y] or self.board[x, y] != -2:
                return
            visited[x, y] = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Movimentos possíveis: direita, baixo, esquerda, cima
            for dx, dy in directions:
                dfs(x + dx, y + dy)

        # Começa a DFS a partir da primeira célula de água encontrada
        dfs(water_cells[0][0], water_cells[0][1])

        # Verifica se todas as células de água foram visitadas
        return np.all(visited[water_cells[:, 0], water_cells[:, 1]])
    
    def tamanho_island_sizes(self):
        """
        Verifica se as ilhas possuem o tamanho correto de conexões
        :return: True se uma ilha principal (valor >=1) estiver conectada com ilhas (-1) na quantidade correta
        """
        
        visited = np.zeros((self.size, self.size), dtype=bool)

        def dfs(x, y, target):
            if x < 0 or x >= self.size or y < 0 or y >= self.size or visited[x, y] or (self.board[x, y] != -1 and self.board[x, y] != target):
                return 0
            visited[x, y] = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            count = 1
            for dx, dy in directions:
                count += dfs(x + dx, y + dy, target)
                return count

        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] >= 1 and not visited[x, y]:
                    island_size = self.board[x, y]
                    counted_size = dfs(x, y, island_size)
                    if counted_size != island_size:
                        return False
        return True

    def islands_separated_correctly(self):
        '''
        A ideia é selecionar um valor numerado (ilhas com valor >=1 e percorrer em profundidade até encontrar valores diferentes
        de -1, caso encontre -2 aquele ramo está terminado. Caso encontre outro valor >=1, isso indica que existem ilhas conectadas
        '''
        visited = np.zeros((self.size, self.size), dtype=bool)

        def dfs(x, y, island_value):
            if x < 0 or x >= self.size or y < 0 or y >= self.size or visited[x, y] or self.board[x, y] == -2:
                return False
            if self.board[x, y] >= 1 and self.board[x, y] != island_value:
                return True
            visited[x, y] = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            return any(dfs(x + dx, y + dy, island_value) for dx, dy in directions)

        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] >= 1 and not visited[x, y]:
                    if dfs(x, y, self.board[x, y]):
                        return False
        return True

    def calcula_islands_and_water(self):
        ''' 
        Verifica se o valor de mares é igual a diferença entre o tamanho do tabuleiro - o número de ilhas (para evitar ilhas soltas)
        :return: True é a proporção estiver correta, False se tiver ilhas soltas
        ''' 
        total_cells = self.size * self.size
        num_islands = 0
        num_water = 0

        for row in self.board:
            for cell in row:
                if cell >= 1:
                    num_islands += cell
                elif cell == -2:
                    num_water += 1

        # Verifica se a condição é satisfeita
        if total_cells - num_islands == num_water:
            return True
        else:
            return False
    
    def no_vazio_spaces(self):
        '''
        Verifica se o tabuleiro está completamente preenchido, sem espaços vazios (valores 0).
        :return: True se não houver espaços vazios, False caso contrário.
        '''
        for row in self.board:
            if 0 in row:  # Verifica se há algum 0 na linha
                return False
        return True
    
    def fitness_function(self, solution):
        size = int(len(solution) ** 0.5)
        fitness = 0
        board = np.array(solution).reshape((size, size))
        #print("Solução ==> ", solution)
        # tirei daqui
        
        # game = NurikabeGame(board)   ==>  chamava quando executava
    
        # Avaliando a solução com base nos critérios do jogo
    
        # Penalidade se houver quadrados 2x2 de mar
        if self.check_block_pools():
            fitness -= 1
            
        # Penalidade se a água não estiver conectada
        if self.verifica_water_connected():
            fitness -= 2
        
        # Penalidade se o tamanho das ilhas estiver incorreto
        if not self.tamanho_island_sizes():
            fitness -= 3
    
        # Penalidade se as ilhas não estiverem separadas corretamente
        if not self.islands_separated_correctly():
            fitness -= 4
        else:
            fitness += 4
        # Penalidade se a proporção de ilhas e mar estiver incorreta
        if not self.calcula_islands_and_water():
            fitness -= 5
        else:
            fitness += 5
    
        # Penalidade se houver espaços vazios
        if not self.no_vazio_spaces():
            fitness -= 6
    
        # Adicionar recompensas para soluções parcialmente corretas
        # Recompensa para cada ilha com tamanho correto
        if self.tamanho_island_sizes():
            fitness += 3  # Recompensa para cada ilha com tamanho correto

        # Recompensa para cada célula de água conectada
        if self.verifica_water_connected():
            fitness += 2  # Recompensa para cada célula de água conectada
        
        return fitness
    
   # def generate_initial_population(self, size, puzzle):
    #    population = []
     #   for _ in range(size):
      #      individual = np.copy(puzzle).flatten() # retorna uma copia colapsada em uma dimensao
       #     individual = np.copy(puzzle)
         #   for idx, value in enumerate(individual):
          #      if value == 0:
          #          individual[idx] = np.random.choice([-2, -1])
           # population.append(individual)
        # return population
    
    def generate_initial_population(self, size, puzzle):
        population = []
    
        def fill_island(individual, x, y, count):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < individual.shape[0] and 0 <= ny < individual.shape[1] and count > 0:
                    if individual[nx, ny] == 0:
                        individual[nx, ny] = -1
                        count -= 1
                    elif individual[nx, ny] == -1:
                        count -= 1

        for _ in range(size):
            individual = np.copy(puzzle)
            for x in range(individual.shape[0]):
                for y in range(individual.shape[1]):
                    if individual[x, y] > 0:  # Se a célula contém um valor positivo
                        fill_island(individual, x, y, individual[x, y])
        
            # Preenchendo os espaços vazios restantes
            for x in range(individual.shape[0]):
                for y in range(individual.shape[1]):
                    if individual[x, y] == 0:
                        individual[x, y] = np.random.choice([-2, -1])
        
            population.append(individual)
    
        return population

    def genetico(self):
        # Parâmetros do algoritmo genético
        algoritmo_param = {
            'max_num_iteration': None,
            'population_size': 100,
            'mutation_probability': 0.05,
            'elit_ratio': 0.01,
            'crossover_probability': 0.1,
            'parents_portion': 0.4,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': 1000  # máximo de iterações sem melhorar
        }
        
                
        # Converter o puzzle para uma matriz numpy
        puzzle_array = np.array(self.puzzle)
        total_cells = puzzle_array.size  # Número total de células
        ### -------------
        # Convertendo a solução em uma matriz de acordo com os valores esperados
        #for i in range(size):
        #    for j in range(size):
        #        if board[i, j] == 0:
        #            board[i, j] = -2  # Mar
        #        elif board[i, j] == 1:
        #            board[i, j] = -1  # Ilha
        ### -------------
        print("Vai executar o modelo")
        print("Modelo ficou assim: ", puzzle_array.shape)

        # Define os limites das variáveis baseados na matriz original
        variable_boundaries = []
        for value in puzzle_array.flatten():
            if value > 0:  # Manter valores positivos
                variable_boundaries.append([value, value])
            elif value == 0:  # Espaços modificáveis
                variable_boundaries.append([-2, -1])
            else:  # Outros valores (caso existam)
                variable_boundaries.append([value, value])
        variable_boundaries = np.array(variable_boundaries)
        print("Variáveis Limites = ", variable_boundaries)
        
        
        def initial_population():
            return self.generate_initial_population(algoritmo_param['population_size'], puzzle_array)

        # Modelo definido
        model = ga(
            function=self.fitness_function,
            dimension=total_cells,
            variable_type='int',
            variable_boundaries=variable_boundaries,
            algorithm_parameters=algoritmo_param
        )
        
        model.run()
        convergence = np.array(model.report)
        solution =  np.array(model.output_dict)
        solution.ndim
        solution.shape
        print("Relatório da convergência: ", convergence)

        if 'variable' in solution:
            gerado = np.array(solution['variable']).reshape((self.size, self.size))
            print("Relatório da convergência: ", convergence)

            # Verifique se a chave 'variable' está presente no dicionário output_dict
            if 'variable' in solution:
                solu = solution['variable']
                print("A solução parcial ==>")
                print_board(np.array(solu).reshape((self.size, self.size)))
            else:
                print("Erro: 'variable' não encontrado em model.output_dict")

            if 'variable' in solution:
                gerado = np.array(solution['variable']).reshape((self.size, self.size))
                print("Solução gerada:")
                print_board(gerado)
            else:
                print("Erro: 'solution' não encontrado em model.output_dict")
        else:
            print("Erro: 'variable' não encontrado em model.output_dict")

        return solution
        

