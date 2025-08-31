import numpy as np

class Judge():
    # Não tem como finalizar o jogo, só acaba quando terminam as epochs
    #  r_search > r_wait
    # Tanto faz se ele encontrar ou não na busca, ele recebe a recompensa  de r_wait 
    def __init__(self,player,  r_search: float, r_wait: float, alpha: float, beta:float):
        self.player = player # só tem um jogador 
        self.r_search = r_search
        self.r_wait = r_wait
        self.alpha = alpha
        self.beta = beta
        
        
    def reset(self):
        self.player.reset()
        
        
    
    def transition(self, state, action ):
        # Vai ser 1 se tiver high 0 se tiver low
        charge_level = state.charge_level
        
        #  Se ta high
        if charge_level == 1: