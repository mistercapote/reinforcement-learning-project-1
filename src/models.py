import numpy as np

class Action:
    def __init__(self, action: str):
        self.action = action
    
    def __eq__(self, other):
        return self.action == other.action
    
    def __hash__(self):
        return hash(self.action)

    def __str__(self):
        return self.action
    
    def __repr__(self):
        return self.action
    
class State:
    def __init__(self, charge_level: int, *actions: Action):
        self.charge_level = charge_level
        self.actions = actions
    
    def __eq__(self, other):
        return self.charge_level == other.charge_level
    
    def __hash__(self):
        return hash(self.charge_level)
    
    def __str__(self):
        return f"State(charge_level={self.charge_level})"
    
    def __repr__(self):
        return f"State(charge_level={self.charge_level})"
    
    def get_actions(self):
        return self.actions
    
class Game():
    
    search = Action('search')
    wait = Action('wait')
    recharge = Action('recharge')
    high = State(1, search, wait)
    low = State(0, search, wait, recharge)
    
    def __init__(self, player,  r_search: float, r_wait: float, alpha: float, beta: float):
        
        self.player = player # só tem um jogador 
        # Invariante: r_search > r_wait
        self.r_search = r_search
        self.r_wait = r_wait
        self.alpha = alpha
        self.beta = beta
        
    def reset(self):
        """Reseta a bateria e estados do jogador (robozinho) para situação inicial 
        """
        self.player.reset()
    
    def transition(self, state: State, action: Action) -> tuple[State, int]:
        """Realiza a ação  atualiza o estado do jogador (robozinho).
        Não tem como finalizar o jogo, só acaba quando terminam as epochs

        Args:
            state (State): estado atual da bateria do player (robozinho)
            action (Action): ação atual do player

        Returns:
            tuple[State, int]: o novo estado e a recompensa da ação
        """
        new_state = None
        reward = 0
        # Se ele busca
        if action == self.search:
            reward = self.r_search
            # Se ta com bateria alta
            if state == self.high:
                # Tanto faz se ele encontrar ou não na busca, ele recebe a recompensa  de r_wait  
                if np.random.rand() < self.alpha: 
                    new_state = self.high
                else: 
                    new_state = self.low
            # Se ta com bateria baixa
            else:
                if np.random.rand() < self.beta:
                    new_state = self.low
                else:
                    new_state = self.high
                    reward = -3
        # Se ele espera
        elif action == self.wait:
            new_state = state
            reward = self.r_wait
        # Se ele recarrega
        elif action == self.recharge:
            new_state = self.low
            reward = 0
        return new_state, reward
                
class Player:
    def __init__(self, step_size: float, epsilon: float):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.estimations[Game.high] = 0.5
        self.estimations[Game.low] = 0.5
        self.action_counts = {
            Game.high: {Game.search: 0, Game.wait: 0},
            Game.low: {Game.search: 0, Game.wait: 0, Game.recharge: 0}
        }
        
    def reset(self):
        self.states = []
        self.greedy = []

    def add_state(self, state: State):
        self.states.append(state)

        
    def backup(self, reward: int, new_state: State):
        if len(self.states) < 2:
            return  # Não há estado anterior para fazer backup
        td_error = reward + self.estimations[new_state] - self.estimations[self.states[-2]]
        self.estimations[self.states[-2]] += self.step_size * td_error

    def set_action(self) -> Action:
        state = self.states[-1]
        actions = state.get_actions()
        
        if np.random.rand() < self.epsilon:
            action = np.random.choice(actions)
        else:
            values = []
            for a in actions:
                if a == Game.search:
                    value = max(self.estimations[Game.high], self.estimations[Game.low])
                    values.append((value, a))
                elif a == Game.wait:
                    value = self.estimations[state]
                    values.append((value, a))
                elif a == Game.recharge:
                    value = self.estimations[Game.high]
                    values.append((value, a))
                
            values.sort(key=lambda x: x[0], reverse=True)
            action = values[0][1]
            
        self.action_counts[state][action] += 1
        return action
    
    def get_policy(self):
        """Retorna a política ótima baseada nas estatísticas"""
        policy = {}
        for state in [Game.high, Game.low]:
            total = sum(self.action_counts[state].values())
            if total > 0:
                policy[state] = {action: count/total for action, count in self.action_counts[state].items()}
            else:
                policy[state] = {action: 0 for action in self.action_counts[state].keys()}
        return policy