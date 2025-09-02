import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
from models import *

def plot_rewards(total_rewards: list[int]) -> None:
    """Plota grafico de linha de recompensa por época"""
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards, label='Recompensa por época')
    plt.xlabel('Época')
    plt.ylabel('Recompensa Total')
    plt.title('Recompensas ao Longo do Treinamento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rewards.png")
    plt.close()

def plot_policy_heatmap(player: Player, game: Game):
    """Plota a política ótima como heatmap"""
    policy = player.get_policy(game)
    
    # Preparar dados para o heatmap
    states = ['High Battery', 'Low Battery']
    actions = ['search', 'wait', 'recharge']
    
    # Criar matriz de probabilidades
    policy_matrix = np.zeros((len(states), len(actions)))
    
    for i, state in enumerate([Game.high, Game.low]):
        for j, action in enumerate([Game.search, Game.wait, Game.recharge]):
            if action in policy[state]:
                policy_matrix[i, j] = policy[state][action]
    
    # Criar DataFrame para melhor visualização
    df = pd.DataFrame(policy_matrix, index=states, columns=actions)
    
    # Plotar heatmap
    plt.figure(figsize=(10, 6))
    
    # Criar colormap personalizado
    cmap = ListedColormap(['#ff6b6b', '#4ecdc4', '#45b7d1'])
    
    sns.heatmap(df, annot=True, cmap=cmap, fmt='.3f', 
                cbar_kws={'label': 'Probabilidade da Ação'})
    
    plt.title('Política Ótima do Robô de Reciclagem\n(Probabilidade de cada ação por estado)')
    plt.xlabel('Ações')
    plt.ylabel('Estado da Bateria')
    
    plt.tight_layout()
    plt.savefig('policy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Imprimir política em formato de tabela
    print("\nPolítica Ótima:")
    print("=" * 50)
    print(f"{'Estado':<15} {'Ação':<10} {'Probabilidade':<15}")
    print("-" * 50)
    for state in states:
        state_obj = Game.high if state == 'High Battery' else Game.low
        for action, prob in policy[state_obj].items():
            print(f"{state:<15} {str(action):<10} {prob:<15.4f}")
    print("=" * 50)
