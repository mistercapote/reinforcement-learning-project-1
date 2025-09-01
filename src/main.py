import numpy as np

class State():
    def __init__(self):
        pass


class Game():
    # Não tem como finalizar o jogo, só acaba quando terminam as epochs
    #  r_search > r_wait
    # Tanto faz se ele encontrar ou não na busca, ele recebe a recompensa  de r_wait 
    def __init__(self,player,  r_search: float, r_wait: float, alpha: float, beta:float):
        """Inicalizada o jogo

        Args:
            player (Player): Jogador (roô)
            r_search (float): recompensa por buscar
            r_wait (float): recompensa por esperar
            alpha (float): probabilidade da bateria permanecer high depois da busca
            beta (float):  probabilidade da bateria permanecer low depois da busca
        """
        self.player = player # só tem um jogador 
        self.r_search = r_search
        self.r_wait = r_wait
        self.alpha = alpha
        self.beta = beta
        self.high = State(1)
        self.low = State(0)
        
    def reset(self):
        """Resetando o jogo
        """
        self.player.reset()
        
    
    def transition(self, state, action):
        # Vai ser 1 se tiver high 0 se tiver low
        
        new_state = None
        # Recompensa
        
        reward = 0 # 5  se for busca e 3 se esperar
        #  Se ta high
        if state == self.high:
            #  Se ele busca
            if action== "search":
                reward = self.r_search
                if np.random.random.random() <= self.alpha:
                    
                    new_state = self.high
                else:
                
                    new_state= self.low
            elif  action == "wait":
                new_state = self.high
                reward = self.r_wait
            
                
        if state ==self.low:
            if action =="search":
                
                if np.random.random.random() <= self.beta:
                    reward = self.r_search
                    new_state = self.low
                else:
                    reward = -3
                    new_state= self.high
            elif action == "wait":
                new_state = self.low
                reward = self.r_wait
            elif action == "recharge":
                new_state = self.high
                reward = 0
        
        return new_state, reward
                
    def reset(self):
        self.player.reset()
                
                
            
        
                
            
                    
                