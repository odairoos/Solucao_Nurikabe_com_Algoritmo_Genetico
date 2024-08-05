'''
Este arquivo possui o teste das funções separadas
'''

def print_board(board):

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
        
# def check_for_pools(board):
#     """
#     Verifica a existência de blocos 2x2 de mar no tabuleiro.
#     Mar é indicado por -2.
    
#     :param board: O tabuleiro do jogo a ser verificado.
#     :return: True se uma "piscina" 2x2 for encontrada, False caso contrário.
#     """
#     rows = len(board)
#     cols = len(board[0]) if rows > 0 else 0

#     # Percorre o tabuleiro, exceto pelas últimas linhas e colunas
#     for x in range(rows - 1):
#         for y in range(cols - 1):
#             # Verifica se todos os 4 quadrados de um bloco 2x2 são mar (-2)
#             if board[x][y] == -2 and \
#                board[x+1][y] == -2 and \
#                board[x][y+1] == -2 and \
#                board[x+1][y+1] == -2:
#                 return True  # Uma "piscina" foi encontrada

#     return False  # Nenhuma "piscina" foi encontrada

# test_board = [
#     [0, -2, -2, 0],
#     [0, -2, -2, 0],
#     [0, 0, 0, -2],
#     [-2, -2, -2, -2]
# ]

# has_pools = check_for_pools(test_board)
# print(f"Existem 'piscinas' no tabuleiro? {'Sim' if has_pools else 'Não'}")



# def dfs(board, x, y, size, start_value, visited):
#     if x < 0 or x >= size or y < 0 or y >= size or visited[x][y]:
#         return False
#     if board[x][y] == 0 or board[x][y] == -2:
#         return False
#     if board[x][y] >= 1 and board[x][y] != start_value:
#         return True  # Encontrou outro número, ou seja, outra ilha principal adjacente.
#     visited[x][y] = True
#     directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#     for dx, dy in directions:
#         if dfs(board, x+dx, y+dy, size, start_value, visited):
#             return True
#     return False

# def are_islands_separated_correctly(board):
#     size = len(board)
#     visited = [[False for _ in range(size)] for _ in range(size)]
#     for x in range(size):
#         for y in range(size):
#             if board[x][y] >= 1 and not visited[x][y]:
#                 if dfs(board, x, y, size, board[x][y], visited):
#                     return False
#     return True

# test_board = [
#     [2, -1,  0,  0,  0],
#     [0,  0,  0, -2, -2],
#     [0,  0,  3, -1,  1],
#     [0, -2, -2, -2, -1],
#     [0,  0,  0,  0, -2]
# ]
# print_board(test_board)
# # Testando a função
# is_separated_correctly = are_islands_separated_correctly(test_board)
# print(f"As ilhas estão separadas corretamente? {'Sim' if is_separated_correctly else 'Não'}")

# def calculate_islands_and_water(board):
#     total_cells = sum(len(row) for row in board)  # Calcula o número total de células no tabuleiro
#     num_islands = 0  # Inicializa a contagem do número de ilhas
#     num_water = 0  # Inicializa a contagem do número de águas
    
#     # Percorre o tabuleiro para calcular o número de ilhas e de águas
#     for row in board:
#         for cell in row:
#             if cell >= 1:
#                 num_islands += cell  # Soma os valores das ilhas
#             elif cell == -2:
#                 num_water += 1  # Conta as células de água
    
#     # Verifica se a diferença entre o número total de células e o número de ilhas é igual ao número de águas
#     if total_cells - num_islands == num_water:
#         return True, num_islands, num_water
#     else:
#         return False, num_islands, num_water

# # Exemplo de uso
# test_board = [
#     [2, -1,  0,  0,  0],
#     [0,  0,  0, -2, -2],
#     [0,  0,  3, 0,  1],
#     [0, -2, -2, -2, -1],
#     [0,  0,  0,  0, -2]
# ]
# print_board(test_board)

# result, num_islands, num_water = calculate_islands_and_water(test_board)
# print(f"O número de ilhas é {num_islands}, e o número de águas é {num_water}.")
# print(f"A configuração do tabuleiro está correta? {'Sim' if result else 'Não'}")


# def check_island_sizes(board):
#     size = len(board)
#     visited = [[False for _ in range(size)] for _ in range(size)]

#     def dfs(x, y, target):
#         if x < 0 or x >= size or y < 0 or y >= size or visited[x][y] or (board[x][y] != -1 and board[x][y] != target):
#             return 0
#         visited[x][y] = True
#         return 1 + dfs(x + 1, y, target) + dfs(x - 1, y, target) + dfs(x, y + 1, target) + dfs(x, y - 1, target)

#     for x in range(size):
#         for y in range(size):
#             if board[x][y] >= 1 and not visited[x][y]:
#                 island_size = board[x][y]
#                 counted_size = dfs(x, y, island_size)
#                 if counted_size != island_size:
#                     return False
#     return True

# # Exemplo de uso
# test_board = [
#     [2, -1, 0, 3, 0],
#     [0,  0, 0, -1,-1],
#     [4,  0, 0,  0, 0],
#     [-1,-1, 0, -1, 0],
#     [-1, 0, 0,  0, 0]
# ]
# print_board(test_board)

# is_correct = check_island_sizes(test_board)
# print(f"O tamanho das ilhas está correto? {'Sim' if is_correct else 'Não'}")

# def is_water_connected(board):
#     size = len(board)
#     visited = [[False for _ in range(size)] for _ in range(size)]
#     water_cells = [(x, y) for x in range(size) for y in range(size) if board[x][y] == -2]

#     if not water_cells:
#         return True  # Se não há água, consideramos como conectada.

#     def dfs(x, y):
#         if x < 0 or x >= size or y < 0 or y >= size or visited[x][y] or board[x][y] != -2:
#             return
#         visited[x][y] = True
#         directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Direita, Baixo, Esquerda, Cima
#         for dx, dy in directions:
#             dfs(x + dx, y + dy)

#     # Inicia a DFS a partir da primeira célula de água encontrada
#     dfs(water_cells[0][0], water_cells[0][1])

#     # Verifica se todas as células de água foram visitadas
#     return all(visited[x][y] for x, y in water_cells)

# #Exemplo de matriz para teste
# test_board = [
#     [-2,  0,  0, -2, -2],
#     [-2, -2,  0,  -2, -2],
#     [ 0,  0,  0, -2, -2],
#     [ 0,  0, -2, -2, -2],
#     [-2, -2, -2,  0,  0]
# ]
# print_board(test_board)

# is_connected = is_water_connected(test_board)
# print(f"Toda a água está conectada? {'Sim' if is_connected else 'Não'}")
# print(is_connected)