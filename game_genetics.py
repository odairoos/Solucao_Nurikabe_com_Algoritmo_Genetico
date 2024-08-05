# Instala a biblioteca para implementação do algoritmo genetico
#!pip install geneticalgorithm
# Ver documentacao pypi.org/project/geneticalgorithm/ 

import numpy as np
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
from game_logic import NurikabeGame
from game_render import print_board


class GeneticsGame:
    # Construtor
    def __init__(self, puzzle, size):
        self.puzzle = puzzle
        self.size = size
        self.board = puzzle
        
    def get_puzzle(self):
        return self.puzzle
    

def funcao_aptidao(self, solution):
        # Avalia a solução usando a lógica do jogo
        game = NurikabeGame(self.size, solution.reshape((self.size, self.size)))
        score = game.evaluate_solution()
        return -score  # Maximizar a pontuação, então minimizamos o negativo dela
   
def gera_individuo(board):
        # Gera uma população inicial aleatória
        population = []
        for _ in range(self.size * self.size):
            individual = np.random.choice([, 1], size=(self.size, self.size))
            population.append(individual)
        return np.array(population)

if __name__ == "__main__":
    ''' puzzle = [
        [0, 2, 0, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4]
    ] '''
    
 # Inicializa o jogo
 gc = GeneticsGame(puzzle,size)
 var_vinculada = gc.get_puzzle()
 print("Tabuleiro inicial:")
 c = input("Continua ?")
 print_board(var_vinculada)
    
 # Define os parâmetros do algoritmo genético
 algoritmo_param = {
        'max_num_iteration': 1000,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 100
    }
    
 # Define os limites das variáveis
 print("Vai definir o limite das variaveis")
 c = input("Continua ?")
 variable_boundaries = np.array([[0, 1]] * (gc.size * gc.size))
 # Cria e executa o modelo do algoritmo genético
    print("Vai executar o modelo")
    model = ga(
        function=gc.funcao_aptidao,
        dimension=gc.size * gc.size,
        variable_type='bool',
        variable_boundaries=variable_boundaries,
        algorithm_parameters=algoritmo_param
    )
                    
    model.run()
    
    # Obtém a melhor solução encontrada
    gerado = model.output_dict['variable'].reshape((gc.size, gc.size))
    print("Solução gerada:")
    print(gerado)
    print("Condição de vitória:", NurikabeGame(gc.size, gerado).check_win_condition())