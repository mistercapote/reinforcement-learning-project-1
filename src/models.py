import numpy as np

class Action:
    def __init__(self, name: str):
        """Inicializa um objeto da classe Action 

        Args:
            name (str): Nome da ação
        """
        self.name = name
    
    def __eq__(self, other)->bool:
        """Compara se dois objetos são iguais 

        Args:
            other (Action): Action a ser comparar

        Returns:
            bool: True se os objetos são iguais, False se são diferentes
        """
        return self.name == other.name
    
    def __hash__(self)->int:
        """Garante que instâncias de ação com o mesmo 
        nome sejam tratadas como a mesma chave em dicionários e conjuntos

        Returns:
            int: identificador do objeto
        """
        return hash(self.name)

    def __str__(self)->str:
        """Representação Legivel da classe Action

        Returns:
            str: Nome legivel do objeto 
        """
        return self.name
    
    def __repr__(self)->str:
        """Representação oficial do objeti

        Returns:
            str: Nome Oficial do objeto
        """
        return self.name
    
    
class State:
    def __init__(self, charge_level: int, *actions: Action):
        """Inicializa um objeto da classe State

        Args:
            charge_level (int): Nivel de Carga da Bateria, 0 se low e 1 se high
        """
        self.charge_level = charge_level
        self.actions = actions
    
    def __eq__(self, other)->bool:
        """Compara se dois objetos são iguais 

        Args:
            other (State): outro objeto a ser comparado

        Returns:
            bool: True se são iguais e False se são diferentes
        """
        return self.charge_level == other.charge_level
    
    def __hash__(self)->int:
        """Garante que instâncias de ação com o mesmo 
        nome sejam tratadas como a mesma chave em dicionários e conjuntos

        Returns:
            int: identificador do objeto
        """
        return hash(self.charge_level)
    
    def __str__(self)->str:
        """Retorna a representação Legivel da classe  state

        Returns:
            str: representação legivel do objeto
        """
        return f"State(charge_level={self.charge_level})"
    
    def __repr__(self)->str:
        """Retorna a representação oficial do objeto

        Returns:
            str:  representação oficial do objeto
        """
        return f"State(charge_level={self.charge_level})"
    
    def get_actions(self)-> tuple[Action]:
        """Retorna as ações disponíveis para o estado atual.

        Returns:
            tuple[Action]: Tupla contendo os objetos Action que podem ser executados
            a partir deste estado.
        """
        
        return self.actions
    
class Player:
    def __init__(self, step_size:float = 0.1, epsilon:float = 0.1, gamma:float = 0.9):
        """Inicializa o jogador com os parâmetros de aprendizado.

        Args:
            step_size (float, optional): A taxa de aprendizado. Defaults to 0.1.
            epsilon (float, optional):  A taxa de exploração para a política
                epsilon-greedy. Defaults to 0.1.
            gamma (float, optional):  O fator de desconto para recompensas
                futuras. Defaults to 0.9.
        """
        self.estimations = {Game.high: 0.5, Game.low: 0.5}
        self.step_size = step_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.was_greedy = False
        self.state = Game.high

    def reset(self):
        """Reseta o estado interno do agente para o início de uma nova época.

        Define o estado inicial como 'high' e reinicia o rastreador de ação
        gananciosa (`was_greedy`).
        """
        self.was_greedy = False
        self.state = Game.high
    
    def act(self, backup):
        """Escolhe uma ação com base na política epsilon-greedy.

        Com probabilidade `epsilon`, escolhe uma ação aleatória dentre as
        possíveis (exploração). Caso contrário, escolhe a ação com o maior
        valor esperado, calculado pela função de `backup` (explotação).

        Args:
            backup: Uma função que calcula o
            valor esperado (retorno) de tomar uma ação em
            um determinado estado.

        Returns:
            Action: A ação selecionada pelo agente.
        """
        if np.random.rand() < self.epsilon:
            self.was_greedy = False
            return np.random.choice(self.state.get_actions())
        
        values = {action: backup(self, self.state, action) for action in self.state.get_actions()}
        highest_value = max(values, key=values.get)
        self.was_greedy = True
        return highest_value
    
    def update(self, next_state: State, reward: int):
        """Atualiza a estimativa de valor do estado usando a regra de TD.

        A atualização só é realizada se a ação anterior foi gananciosa
        (explotação). A função de valor do estado atual é ajustada com base na
        recompensa recebida e na estimativa de valor do próximo estado.

        Args:
            next_state (State): O estado para o qual o agente transicionou.
            reward (int): A recompensa recebida pela transição.
        """
        if self.was_greedy == False:
            return
        
        current_estimation = self.estimations.get(self.state, 0.5)
        next_estimation = self.estimations.get(next_state, 0.5)
        self.estimations[self.state] += self.step_size * (reward + self.gamma * next_estimation - current_estimation)
        self.state = next_state

    def get_policy(self, backup):
        policy = {Game.high: {}, Game.low: {}}
        
        for state in [Game.high, Game.low]:
            possible_actions = state.get_actions()
            
            # Expected values of each action
            values = {action: backup(self, state, action) for action in possible_actions}
            best_value = max(values.values())
            best_actions = [a for a, v in values.items() if v == best_value]
            
            # epsilon-greedy policy
            for action in possible_actions:
                if action in best_actions:
                    prob = (1 - self.epsilon) / len(best_actions) + self.epsilon / len(possible_actions)
                else:
                    prob = self.epsilon / len(possible_actions)
                policy[state][action] = prob
        return policy
 
class Game():
    search = Action('search')
    wait = Action('wait')
    recharge = Action('recharge')
    high = State(1, search, wait)
    low = State(0, search, wait, recharge)
    
    def __init__(self, r_search: float, r_wait: float, alpha: float, beta: float):
        """Inicializa o jogo

        Args:
            r_search (float): Recompesa por buscar
            r_wait (float): Recompesa por procurar
            alpha (float): probabilidade de estando com bateria high permanecer no high (ao afetuar uma busca)
            beta (float):  probabilidade de estando com bateria low permanecer no low (ao efetuar uma busca)

        Raises:
            ValueError: Retorna erro caso r_search for menor ou igual ao r_wait
            ValueError: Retorna erro se alpha ou beta não estão no intervalo [0,1]
        """
        if r_search <= r_wait:
            raise ValueError("Valores de recompensa inválidos. Deve-se respeitar  r_search > r_wait")
        
        if not (0 <= alpha <= 1 and 0 <= beta <= 1):
            raise ValueError("As probabilidades alpha e beta devem estar no intervalo [0, 1].")

        self.r_search = r_search
        self.r_wait = r_wait
        self.alpha = alpha
        self.beta = beta
    
    def transition(self, player: Player) -> int:
        """Realiza a ação  atualiza o estado do jogador (robozinho).
        Não tem como finalizar o jogo, só acaba quando terminam as epochs

        Args:
            state (State): estado atual da bateria do player (robozinho)
            action (Action): ação atual do player

        Returns:
            tuple[State, int]: o novo estado e a recompensa da ação
        """
        state = player.state 
        action = player.act(self.backup)
        
        if state not in [self.low, self.high]:
            raise ValueError("Estado inválido!") 
        if  action not in state.get_actions():
            raise ValueError(f"A ação '{action}' não é permitida no estado '{state}'.")
        
        next_state = None
        reward = 0
        
        # Se ele busca
        if action == self.search:
            reward = self.r_search
            # Se ta com bateria alta
            if state == self.high:
                if np.random.rand() < self.alpha: 
                    next_state = self.high
                else: 
                    next_state = self.low
            # Se ta com bateria baixa
            else:
                if np.random.rand() < self.beta:
                    next_state = self.low
                else:
                    next_state = self.high
                    reward = -3
        # Se ele espera
        elif action == self.wait:
            next_state = state
            reward = self.r_wait
        # Se ele recarrega
        elif action == self.recharge:
            next_state = self.high
            reward = 0
            
        player.update(next_state, reward)
        return reward
    
    def backup(self, player: Player, state: State, action: Action)-> float:
        """Ela aplica a Equação de Bellman para obter uma estimativa do retorno esperado.

        Args:
            player (Player): jogador (robozinho)
            state (State): estado atual
            action (Action): ação

        Returns:
            float: Retorna quão boa é aquela ação naquele estado.
        """
        high = player.estimations.get(self.high, 0.5)
        low = player.estimations.get(self.low, 0.5)
        
        # Return the value depending on wich action and state we're in
        if state == self.high:
            if action == self.search:
                return (self.alpha * (self.r_search + player.gamma * high) + (1-self.alpha)*(self.r_search + player.gamma * low))
            elif action == self.wait:
                return self.r_wait + player.gamma * high
        elif state == self.low:
            if action == self.search:
                return (self.beta * (self.r_search + player.gamma * low) + (1-self.beta)*(-3 + player.gamma * high))
            elif action == self.wait:
                return (self.r_wait + player.gamma * low)
            elif action == self.recharge:
                return (0 + player.gamma * high)
            

            