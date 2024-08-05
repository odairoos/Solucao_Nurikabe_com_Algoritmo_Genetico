'''
Nurikabe raiz
Autor: Rafael Gomes Moreira
e-mail: moreirargm@ita.br
'''
import json
import os
import random

def load_puzzle(size):
    """
    Carrega um puzzle Nurikabe a partir de um arquivo JSON com base no tamanho especificado.
    Assume que o arquivo contém múltiplos objetos JSON válidos separados por quebras de linha.
    
    :param size: Inteiro representando o tamanho do tabuleiro (ex: 5, 7, 10, etc.)
    :return: Uma lista de listas representando o tabuleiro de jogo.
    """
    actual_path = os.getcwd()
    filename = f'sample/{size}x{size}.json'
    puzzles = []

    try:
        with open(filename, 'r') as file:
            for line in file:
                try:
                    puzzle = json.loads(line)
                    puzzles.append(puzzle)
                except json.JSONDecodeError as e:
                    print(f"Erro ao decodificar JSON: {e}")
    except FileNotFoundError:
        print(f'Arquivo {filename} não encontrado.')
        return None

    # Retorna um puzzle aleatório
    import random
    return random.choice(puzzles) if puzzles else None
