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
        
        if r_search <= r_wait:
            raise ValueError("Valores de recompensa inválidos. Deve-se respeitar  r_search > r_wait")
        
        if not (0 <= alpha <= 1 and 0 <= beta <= 1):
            raise ValueError("As probabilidades alpha e beta devem estar no intervalo [0, 1].")

        self.player = player # só tem um jogador 
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
        
        if state not in [self.low, self.high]:
            raise ValueError("Estado inválido!") 
        if  action not in self.next_possible_actions:
            raise ValueError(f"A ação '{action.name}' não é permitida no estado '{state.name}'.")
        
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
    
    def next_possible_actions(self, state: State)-> list[Action]:
        """Retorna as ações possiveis dado o estado atual

        Args:
            state (State): Estado atual

        Returns:
            list[Action]: Lista com as ações possiveis
        """
        possible_actions = []
        if state not in [self.high, self.low]:
            raise ValueError("Estato inválido!")
        
        # Se a bateria estiver  baixa ou cheia, pode procurar ou esperar
        possible_actions.append(self.search)
        possible_actions.append(self.wait)
        
        # Se a bateria estiver baixa, pode procurar,  esperar ou recarregar
        if state== self.low:
            possible_actions.append(self.recharge)
        
        return possible_actions 
        
class Player:

    def __init__(self, step_size = 0.1, epsilon = 0.1, gamma = 0.9):
        self.estimations = {Game.high: 0.5, Game.low: 0.5}
        self.step_size = step_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.was_greedy = False

    # Using the temporal difference method of the tic tac toe example, we'll update the estimations of a state if the decision made is a greedy one, i. e., it wasn't made at random
    def backup(self, state, action, env_params):
        high = self.estimations.get(Game.high, 0.5)
        low = self.estimations.get(Game.low, 0.5)

        alpha = env_params['alpha']
        beta = env_params['beta']
        r_search = env_params['r_search']
        r_wait = env_params['r_wait']

        # Return the value depending on wich action and state we're in
        if state == Game.high:
            if action == Game.search:
                return (alpha*(r_search + self.gamma * high) + (1-alpha)*(r_search + self.gamma * low))
            elif action == Game.wait:
                return r_wait + self.gamma * high
        elif state == Game.low:
            if action == Game.search:
                return (beta*(r_search + self.gama * low) + (1-beta)(-3 + self.gamma*high))
            elif action == Game.wait:
                return (r_wait + self.gamma * low)
            elif action == Game.recharge:
                return (0 + self.gamma * high)
            
    def act(self, state, possible_actions, env_params):

        if np.random.rand() < self.epsilon:
            self.was_greedy = False
            return np.random.choice(possible_actions)
        
        values = {action: self.backup(state, action, env_params) for action in possible_actions}
        highest_value = max(values, key=values.get)
        self.was_greedy = True
        return highest_value
    
    def update(self, state, reward, next_state):
        if self.was_greedy == False:
            return
        
        current_estimation = self.estimations.get(state, 0.5)
        next_estimation = self.estimations.get(next_state, 0.5)
        self.estimations[state] += self.step_size*(reward + self.gamma*next_estimation - current_estimation)