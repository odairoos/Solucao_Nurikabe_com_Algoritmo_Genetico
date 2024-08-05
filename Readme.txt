Nurikabe raiz
Autor: Rafael Gomes Moreira
e-mail: moreirargm@ita.br


Estrutura do projeto:

Requerido: python >= 3.9 (versões anteriores devem funcionar pois somente usei lib padrão)

main.py :: Responsável pela interação com o usuário. 
|
|_ game_logic.py :: Lógica do jogo -> Implementa as regras do Nurikabe.
|_ game_render.py :: Desenha o tabuleiro na tela e a função que pede as jogadas.
|_ sample_reader.py :: Lê os tabuleiros do tamanho escolhido e escolhe um randomicamente.

1) Esta versão foi feita para ser usada no terminal, com o objetivo de facilitar a implementação de heurísticas de resolução.

2) Ao iniciar o jogo (main.py) será consultado qual tamanho de tabuleiro o jogador vai querer jogar. Após isso, um menu perguntará se o jogador quer jogar, ou seja, preencher um campo do tabuleiro, ou verificar a resposta. Se o tabuleiro não estiver completo, o jogador receberá um erro solicitando que o tabuleiro seja preenchido.

3) O tabuleiro é formado por uma matriz (lista de listas):

	a) 0 indica campo vazio
	b) valores >= 1 indica ilhas principais (colocado pelo jogo) e os valores indicam quantas ilhas formam um complexo de ilhas.
	c) -1 indica uma ilha adicionada pelo jogador. Ilha -1 deve ser anexada às ilhas principais para formar complexos de ilhas.
	d) -2 indica o mar.

4) A lógica do jogo encontra-se no arquivo game_logic.py, na classe NurikabeGame e nele estão definidas as seguintes regras:

	a) Se o movimento é valido: Jogador não pode preencher fora do tabuleiro ou sobre uma ilha numerada (>=1)
	b) Verificação de pools 2x2: Não pode haver piscinas 2x2 formada por mar (-2 na matriz)
	c) Verificação se todos os valores de mar (-2) estão conectados. Não pode haver mais de um "grupo" de campos -2 no tabuleiro
	d) O complexo de ilhas, incluindo ilha principal (só pode haver uma ilha principal com valor >=1) e ilhas adicionadas (valor igual a -1) deve possuir a quantidade total igual ao valor da ilha principal.
	e) Os complexos de ilhas (1 ilha principal e n-1 (n= número da ilha principal) conectadas horizontalmente ou verticalmente) não pode tocar outro complexo de ilhas. As ilhas devem ser separadas corretamente.
	f) O valor de mares (-2) deve ser igual a diferença entre a quantidade de espaços do tabuleiro e o número de ilhas. A ideia desta verificação é que, após as demais, se esse valor não for correto, deve indicar que temos ilhas soltas (-1) no tabuleiro.
	g) O tabuleiro deve ser totalmente preenchido com ilhas (-1) ou mar (-2). Para a verificação final, não pode haver espaço vazio (0).

5) Com exceção da verificação de movimentos válidos, todas as regras do Nurikabe somente serão testadas quando a opção de verificar resposta for acionada.

6) As amostras de jogos Nurikabe foram extraídas do site www.puzzle-nurikabe.com usando o crawler de peter hung (phung@post.harvard.edu), com tabuleiros de 5x5 (2000 samples), 7x7(2000 samples), 10x10 (2000 samples), 12x12 (2000 samples), 15x15 (1500 samples) e 20x20 (1500 samples) e as amostras encontram-se no diretório sample.

	