# -*- coding: utf-8 -*-
# Instala a biblioteca para implementação do algoritmo genetico
#!pip install geneticalgorithm
# Ver documentacao pypi.org/project/geneticalgorithm/ 


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
        #self.puzzle = np.array(game.board)
        #self.size = size
        self.puzzle = game
        self.board = game
        self.size = size

    def get_puzzle(self):
        return self.puzzle

    def corrigir(matriz):
        # Encontrar posições onde os valores são  iguais a zero
        valor_zero = np.argwhere(matriz == 0)
        # Gerar valores aleatórios entre -1 e 1 nas posições encontradas
        alternador = -1
        # Substituir os valores não positivos pelos valores aleatórios
        for indice in valor_zero:
            matriz[tupla(indice)] = alternador
            alternador *= -1
        return matriz 

    # Função objetivo: precisa respeitar as restrições
    def funcao_aptidao(self, puzzle):
        # Avalia a solução usando a lógica do jogo
        fa = NurikabeGame.check_win_condition(puzzle)
        game = NurikabeGame(self.size, puzzle.reshape((self.size, self.size)))
        score = game.evaluate_solution()
        return -score  # Maximizar a pontuação, então minimizamos o negativo dela

        # Define os parâmetros do algoritmo genétic
        
    def genetico(self):
        # poderia ser incluindo GEN_NUM = ? (numero de genes), SIGMA = 0.3 (limite de mutação)
        
        algoritmo_param = {
                'max_num_iteration': 1000,
                'population_size': 10000,
                'mutation_probability': 0.1,
                'elit_ratio': 0.01,
                'crossover_probability': 0.5,
                'parents_portion': 0.3,
                'crossover_type': 'uniform',
                'max_iteration_without_improv': 100  # máximo de iterações sem melhorar
                }
        print("GENETICO ATUANDO AGORA!!")
        #print_board(seboard)
        #variable_boundaries = np.array([[0, 1]] * (self.size * self.size
        
        jogo = self.puzzle
        # Cria e executa o modelo do algoritmo genético
        print("Vai executar o modelo")
    
    
        # variable_boundaries = limites das variaveis: matriz que indica os limites inferior e superior
        # indice 0 vai de -1 a 1
        # indice 1 vai de -1 a 1
        model = ga(
                function=self.funcao_aptidao,
                #dimension=self.size * self.size,
                dimension=2,
                variable_type='int',
                variable_boundaries=np.array([[-1,1], [-1,1]]),
                algorithm_parameters=algoritmo_param
                )
        ## Executa o modelo
        model.run()

        # Obtém a melhor solução encontrada
        gerado = model.output_dict['variable'].reshape((self.size, self.size))
        print("Solução gerada:")
        print_board(gerado)
        ng = NurikabeGame(self.size, gerado)
        pool = ng.check_for_pools()
        if pool:
            gerado = corrigir()
            print("Matriz após ajuste de pools:")
            print_board(gerado)
            water = ng.water_connected()
        if water:
            gerado = corrigir()
            print("Matriz após ajuste de conexão de água:")
            print_board(gerado)
        return