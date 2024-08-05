# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:23:37 2024

@author: Odair
"""
import tkinter as tk
from tkinter import ttk
from game_logic import NurikabeGame
from game_render import print_board
from game_genetics import GeneticsGame

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.lb_opcoes = tk.Label(self, text="Escolha sua opção")
        self.lb_opcao1 = tk.Label(self, text="Opcao 1: JOGAR")
        self.lb_opcao2 = tk.Label(self, text="Opcao 2: VERIFICAR")
        self.lb_opcao3 = tk.Label(self, text="Opcao 3: SISTEMA JOGAR")
        
        self.lb_esc = tk.Label(self, text="Qual ?")
        self.v_esc = tk.Entry(self)
        
        self.tv = ttk.Treeview(self, columns=('1', '2', '3', '4', '5'), show='headings')
        for col in ('1', '2', '3', '4', '5'):
            self.tv.column(col, minwidth=0, width=50)
            self.tv.heading(col, text=col)

        self.bt_ok = tk.Button(self, text="Ok", command=self.executar)
        self.lb_game_msg = tk.Label(self, text="")
        self.lb_error_msg = tk.Label(self, text="")
        
        
        # Positioning widgets in grid
        self.lb_opcoes.grid(column=1, row=0, sticky='w')
        self.lb_opcao1.grid(column=1, row=1, sticky='w')
        self.lb_opcao2.grid(column=1, row=2, sticky='w')
        self.lb_opcao3.grid(column=1, row=3, sticky='w')
        self.lb_esc.grid(column=1, row=4, sticky='w')
        self.v_esc.grid(column=2, row=4, sticky='w')
        self.bt_ok.grid(column=2, row=5, sticky='w')
        self.tv.grid(column=0, row=6, columnspan=5, pady=5)
        self.lb_game_msg.grid(column=0, row=7, columnspan=5, pady=5)
        self.lb_error_msg.grid(column=0, row=8, columnspan=5, pady=5)

    def executar(self):
        try:
            v_esc = int(self.v_esc.get())
            if v_esc == 1:
                x, y, value = ask_for_move()
                if self.game.make_move(x, y, value):
                    print("Movimento realizado.")
                else:
                    print("Movimento inválido. Por favor, tente novamente.")
            elif v_esc == 2:
                if self.game.check_win_condition():
                    self.lb_game_msg.config(text="Parabéns! Você completou o Nurikabe.")
                else:
                    self.lb_game_msg.config(text="A solução ainda não está correta. Keep trying!!!! kkk")
            elif v_esc == 3:
                print("Vou chamar o Genetico")
                best_solution = self.gameG.run_genetic_algorithm()
                print_board(best_solution)
                self.update_treeview(best_solution)
                if NurikabeGame(self.gameG.size, best_solution).check_win_condition():
                    self.lb_game_msg.config(text="O algoritmo encontrou uma solução válida!")
                else:
                    self.lb_game_msg.config(text="O algoritmo não encontrou uma solução válida.")
            else:
                self.lb_error_msg.config(text="Opção inválida. Escolha 1, 2 ou 3.")
        except ValueError:
            self.lb_error_msg.config(text="Por favor, insira um número válido.")

    
    def update_treeview(self, solution):
        # Clear the treeview
        for item in self.tv.get_children():
            self.tv.delete(item)
        
        # Insert the new solution into the treeview
        for row in solution:
            self.tv.insert('', 'end', values=tuple(row))
            
if __name__ == "__main__":
    app = tk.Tk()
    app.title("NURIKABE - VERSÃO GRÁFICA")
    app.geometry("1000x500")
    myapp = App(master=app)
    myapp.mainloop()
