'''
Nurikabe raiz
Autor: Rafael Gomes Moreira
e-mail: moreirargm@ita.br
'''

import copy
import os
import time
import numpy as np
from sample_reader import load_puzzle
from game_logic import NurikabeGame
from game_render import print_board, ask_for_move
#from game_genetics import GeneticsGame
#from game_genetico4 import Genetics_Game 
#from game_genetico5 import Genetics_Game
#from game_genetico6 import Genetics_Game
from game_genetico7 import Genetics_Game

def clear_screen():
    '''
    Limpa a tela para evitar a poluição no terminal
    '''
    # print(os.name)
    os.system('cls' if os.name == 'nt' else 'clear')
    
def show_menu():
    '''
    Exibe um menu com as opções para o usuário
    '''
    print("\nEscolha uma opção:")
    print("1 - Jogar")
    print("2 - Verificar resposta")
    print("3 - Algoritmo jogar")
    print("9 - Sair")
    choice = input("Digite sua escolha (1, 2 ou 3): ")
    #choice = 3
    return choice
    


def main():
    #clear_screen()
    print("Bem-vindo ao Nurikabe, Odair!")
   
    # Escolha do tamanho do tabuleiro
    # size = int(input("Escolha o tamanho do tabuleiro (5, 7, 10, 12, 15, 20): "))
    size = 20
    puzzle_sample = load_puzzle(size)
    
    if puzzle_sample is None:
        print("Não foi possível carregar o puzzle. Por favor, tente novamente.")
        return
    
    puzzle = copy.deepcopy(puzzle_sample)
    
    game = NurikabeGame(puzzle)
    gameG = Genetics_Game(game, size)
    start_time = time.time() #inicializar um cronômetro
    
    while True:
        clear_screen()
        print_board(game.board)
        choice = show_menu()
        
        if choice == '1':
            x, y, value = ask_for_move()
            
            if game.make_move(x, y, value):
                print("Movimento realizado.")
            else:
                print("Movimento inválido. Por favor, tente novamente.")
        
        elif choice == '2':
            if game.check_win_condition():
                end_time = time.time() #final do cronômetro
                elapsed_time = end_time - start_time
                print(f'Parabéns! Você completou o Nurikabe em {elapsed_time:.2f} segundos.')
                time.sleep(3)
                break
            else:
                print('A solução ainda não está correta. Keep trying!!!! kkk')
                time.sleep(3)
        elif choice == '3':
                #clear_screen()
                print("O genetico vai jogar")
                print_board(game.board)
                resolvido, best_score = gameG.genetico()
                print("Fim de Jogo!")
                print("Melhor solução encontrada ==>> ", resolvido)
                # print("Score ", score)
                print("Máximo score = 21, Melhor score alcançado ==> ", np.max(best_score))
                print_board(resolvido)
                end_time = time.time() #final do cronômetro
                elapsed_time = end_time - start_time
                print(f'Nurikabe completado em {elapsed_time:.2f} segundos.')
                time.sleep(36)
        elif choice == '9':
                return false
        else:
            print('Opção inválida. Escolha 1, 2 ou 3.') 

if __name__ == "__main__":
    main()
