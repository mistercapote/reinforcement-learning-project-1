import numpy as np
from models import *
from plot import *

def train(epochs: int, steps_per_epoch: int, r_search: float, r_wait: float, alpha: float, beta: float):
    player = Player(step_size=0.1, epsilon=0.1, gamma=0.9)
    game = Game(r_search, r_wait, alpha, beta)
    total_rewards = []
    
    for i in range(1, epochs+1):
        player.reset()
        episode_reward = sum(game.transition(player) for _ in range(steps_per_epoch))            
        total_rewards.append(episode_reward)
        
        if i % 100:
            print(f'\rEpoch {i} de {epochs} concluída. Recompensa: {episode_reward} | high={player.estimations[Game.high]:.4f}, low={player.estimations[Game.low]:.4f}', end='', flush=True)

    with open('rewards.txt', 'w') as f:
        for i, reward in enumerate(total_rewards, 1):
            f.write(f'Epoch {i}: {reward}\n')
    
    return player, game, total_rewards

def main():
    np.random.seed(42)
    EPOCHS = 10000
    STEPS_PER_EPOCH = 1000
    R_SEARCH = 5
    R_WAIT = 1
    ALPHA = 0.7
    BETA = 0.6
    
    print("Iniciando treinamento do Robô de Reciclagem...")
    print(f"Parâmetros: R_search={R_SEARCH}, R_wait={R_WAIT}, alpha={ALPHA}, beta={BETA}")
    print(f"Épocas: {EPOCHS}, Passos por época: {STEPS_PER_EPOCH}")
    print("-" * 60)
    
    player, game, total_rewards = train(
        epochs=EPOCHS, 
        steps_per_epoch=STEPS_PER_EPOCH,
        r_search=R_SEARCH,
        r_wait=R_WAIT,
        alpha=ALPHA,
        beta=BETA
    )
    
    print("\n" + "-" * 60)
    print("\nTreinamento concluído!")
    print(f"Recompensa média final: {np.mean(total_rewards[-1000:]):.2f}")
    print(f"Estimativas finais - High: {player.estimations[Game.high]:.4f}, Low: {player.estimations[Game.low]:.4f}")
    
    # Plotar gráficos
    print("\nGerando gráficos...")
    plot_rewards(total_rewards)
    plot_policy_heatmap(player, game.backup)
    
    print("Arquivos salvos:")
    print("- rewards.txt: Recompensas de cada época")
    print("- rewards_plot.png: Gráfico das recompensas")
    print("- policy_heatmap.png: Heatmap da política ótima")
    
if __name__ == '__main__':
    main()