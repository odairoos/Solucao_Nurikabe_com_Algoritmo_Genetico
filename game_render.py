'''
Nurikabe raiz
Autor: Rafael Gomes Moreira
e-mail: moreirargm@ita.br

Este arquivo contém a lógica do jogo
'''



def print_board(board):
    """
    Imprime o tabuleiro do jogo no console com numeração de linhas e colunas,
    substituindo 0 por espaço vazio, -1 por um ponto no centro, e -2 pelo
    símbolo ASCII de valor decimal 219 para o fluxo de água.
    :param board: O tabuleiro do jogo a ser impresso.
    """
    # Símbolo ASCII para fluxo de água
    full_block = '#'
    
    # Cabeçalho com números de colunas
    header = '    ' + '   '.join(str(i + 1) for i in range(len(board)))
    print(header)

    # Linha de separação inicial
    border = '  +' + '---+' * len(board)
    print(border)

    # Imprimir cada linha do tabuleiro com numeração e separadores
    for i, row in enumerate(board):
        row_str = f"{i + 1} | " + " | ".join(' ' if cell == 0 else '.' if cell == -1 else full_block if cell == -2 else str(cell) for cell in row) + ' |'
        print(row_str)
        print(border)

def ask_for_move():
    """
    Pede ao usuário para fazer um movimento.
    :return: Uma tupla (x, y, valor) representando o movimento do usuário.
    """
    try:
        x = int(input('Digite a linha: '))
        y = int(input('Digite a coluna: '))
        value_input = int(input('Digite o valor (1 para Ilha, 2 para Água): '))
        value = -1 if value_input == 1 else -2
        return x - 1, y - 1, value  # Ajuste para base 0
    except ValueError:
        print("Por favor, insira valores válidos.")
        return ask_for_move()
