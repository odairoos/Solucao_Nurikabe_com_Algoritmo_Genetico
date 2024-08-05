# game_logic.py

'''
Nurikabe raiz
Autor: Rafael Gomes Moreira
e-mail: moreirargm@ita.br

Este arquivo contém a lógica do jogo
'''

class NurikabeGame:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.board = puzzle 
        self.size = len(puzzle)

    def print_board(self):
        for row in self.board:
            print(' '.join(str(cell) for cell in row))
        print()

    def is_move_valid(self, x, y, value):
        """
        Verifica se um movimento é válido.
        :param x: Linha do tabuleiro.
        :param y: Coluna do tabuleiro.
        :param value: Valor a ser colocado no tabuleiro (1 para ilha, 2 para mar).
        :return: True se o movimento for válido, False caso contrário.
        """
        if 0 <= x < self.size and 0 <= y < self.size:
            #return self.board[x][y] == 0
            return self.board[x][y] in [0, -1, -2]
        return False
    
    def make_move(self, x, y, value):
        """
        Realiza um movimento no jogo após validação.
        :param x: Linha do tabuleiro.
        :param y: Coluna do tabuleiro.
        :param value: Valor a ser colocado no tabuleiro (1 para ilha, 2 para mar).
        """
        if self.is_move_valid(x, y, value):
            self.board[x][y] = value
            return True
        return False
    
    # A partir daqui são as verificações conforme as regras do Nurikabe
    
    def check_for_pools(self):
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
    
    def is_water_connected(self):
        """
        Verifica se toda a água no tabuleiro está conectada.
        :return: True se toda a água estiver conectada, False caso contrário.
        """
        visited = [[False for _ in range(self.size)] for _ in range(self.size)]
        water_cells = [(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == -2]

        if not water_cells:
            return True  # Se não há água, consideramos como conectada.

        # Função auxiliar para realizar a DFS a partir de uma célula de água
        def dfs(x, y):
            if x < 0 or x >= self.size or y < 0 or y >= self.size or visited[x][y] or self.board[x][y] != -2:
                return
            visited[x][y] = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Movimentos possíveis: direita, baixo, esquerda, cima
            for dx, dy in directions:
                dfs(x + dx, y + dy)

        # Começa a DFS a partir da primeira célula de água encontrada
        dfs(water_cells[0][0], water_cells[0][1])

        # Verifica se todas as células de água foram visitadas
        return all(visited[x][y] for x, y in water_cells)
    
    def check_island_sizes(self):
        '''
        Verifica se as ilhas possuem o tamanho correto de conexões
        :return: True se uma ilha principal (valor >=1) estiver conectada com ilhas (-1) na quantidade correta
        '''
        visited = [[False for _ in range(self.size)] for _ in range(self.size)]
        def dfs(x, y, target):
            if x < 0 or x >= self.size or y < 0 or y >= self.size or visited[x][y] or (self.board[x][y] != -1 and self.board[x][y] != target):
                return 0
            visited[x][y] = True
            return 1 + sum(dfs(x + dx, y + dy, target) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)])

        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] >= 1 and not visited[x][y]:
                    island_size = self.board[x][y]
                    counted_size = dfs(x, y, island_size)
                    if counted_size != island_size:
                        return False
        return True
  
    
    def dfs(self, x, y, start_value, visited):
        '''
        A ideia desse dfs é selecionar um valor numerado (ilhas com valor >=1 e percorrer em profundidade até encontrar valores diferentes
        de -1, caso encontre -2 aquele ramo está terminado. Caso encontre outro valor >=1, isso indica que existem ilhas conectadas
        '''
        if x < 0 or x >= self.size or y < 0 or y >= self.size or visited[x][y]:
            return False
        if self.board[x][y] == 0 or self.board[x][y] == -2:
            return False
        if self.board[x][y] >= 1 and self.board[x][y] != start_value:
            return True  # Encontrou outro número, ou seja, outra ilha principal adjacente.
        visited[x][y] = True
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            if self.dfs(x + dx, y + dy, start_value, visited):
                return True
        return False

    def are_islands_separated_correctly(self):
        '''
        A ideia é selecionar um valor numerado (ilhas com valor >=1 e percorrer em profundidade até encontrar valores diferentes
        de -1, caso encontre -2 aquele ramo está terminado. Caso encontre outro valor >=1, isso indica que existem ilhas conectadas
        '''
        visited = [[False for _ in range(self.size)] for _ in range(self.size)]
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] >= 1 and not visited[x][y]:
                    if self.dfs(x, y, self.board[x][y], visited):
                        return False
        return True
    
    def calculate_islands_and_water(self):
        ''' 
        verifica se o valor de mares é igual a diferença entre o tamanho do tabuleiro - o número de ilhas (para evitar ilhas soltas)
        :return: True é a proporção estiver correta, false se tiver ilhas soltas
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
    
    def no_empty_spaces(self):
        '''
        Verifica se o tabuleiro está completamente preenchido, sem espaços vazios (valores 0).
        :return: True se não houver espaços vazios, False caso contrário.
        '''
        for row in self.board:
            if 0 in row:  # Verifica se há algum 0 na linha
                return False
        return True
    
    # Verificação se você venceu o jogo
        
    def check_win_condition(self):
        '''
        Verifica a condição de vitória do jogo.
        :return: True se o jogador vencer, False caso contrário.
        '''
        if not self.no_empty_spaces():
            print("Há espaços vazios no tabuleiro.")
            return False
        if not self.check_island_sizes():
            print("O tamanho de uma ou mais ilhas está incorreto.")
            return False
        if not self.is_water_connected():
            print("O mar não está totalmente conectado.")
            return False
        if self.check_for_pools():
            print("Existem 'piscinas' de água 2x2.")
            return False
        if not self.are_islands_separated_correctly():
            print("Algumas ilhas não estão separadas corretamente.")
            return False
        print("Parabéns! Você resolveu o puzzle corretamente.")
        return True

    def check_win_spaces(self):
        '''
        Verifica a condição de vitória do jogo.
        :return: True se o jogador vencer, False caso contrário.
        '''
        if not self.no_empty_spaces():
            # print("Há espaços vazios no tabuleiro.")
            return -1
        
    def check_win_island(self):  
        if not self.check_island_sizes():
            print("O tamanho de uma ou mais ilhas está incorreto.")
            return -2
    def check_win_water(self): 
        if not self.is_water_connected():
           # print("O mar não está totalmente conectado.")
            return -1
        

   
