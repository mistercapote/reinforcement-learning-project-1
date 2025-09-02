# Aprendizado por Reforço para Robô de Reciclagem

## Introdução

Este relatório descreve a implementação de um algoritmo de Aprendizado por Reforço, baseado em *Temporal Difference* (TD), para resolver o problema do Robô de Reciclagem, (Ex. 3.3 de Sutton & Barto - "Reinforcement Learning: An Introduction"). 

O agente aprende valores de estado e age com política *ε-greedy* com *one-step lookahead* usando o modelo conhecido (α, β e recompensas) para escolher ações. O objetivo é **maximizar a recompensa média** em uma tarefa contínua (desconto γ).


## Equipe

**Professor orientador:**
- Flávio Codeço Coelho

**Alunos participantes:**
- Jaime Willian Carneiro
- Luiz Eduardo Bravin
- Walleria Simões Correia

## Requisitos

Para instalar as bibliotecas necessárias, execute:
```bash
  pip install -r requirements.txt
```
Mínimo recomendado: `numpy`, `matplotlib`, `seaborn`, `pandas`.

## 1. Descrição do Problema (MDP)

O problema é modelado como um Processo de Decisão de Markov (MDP) finito, definido pelos seguintes componentes:

### Estados (S)

O robô pode estar em um de dois estados, baseados no seu nível de bateria:
* `high`: Nível de bateria alto.
* `low`: Nível de bateria baixo.

### Ações (A)

O conjunto de ações disponíveis depende do estado atual:
* `A(high)` = {`search`, `wait`}
* `A(low)` = {`search`, `wait`, `recharge`}

### Modelo de Transição e Recompensa

O modelo de transição e recompensa foi implementado conforme a descrição do livro-texto. As transições são probabilísticas e dependem dos seguintes parâmetros, que foram definidos para esta implementação:

* **`α`**: Probabilidade de a bateria permanecer alta após uma busca, começando com bateria alta: `high` $\rightarrow$ `search` $\rightarrow$ `high`.
* **`β`**: Probabilidade de a bateria permanecer baixa após uma busca, começando com bateria baixa: `low` $\rightarrow$ `search` $\rightarrow$ `low`.
* **`r_search`**: Recompensa esperada ao executar a ação `search`.
* **`r_wait`**: Recompensa esperada ao executar a ação `wait`.

A tabela a seguir, adaptada do livro-texto, resume a dinâmica do ambiente com os parâmetros escolhidos:

| Estado inicial | Ação | Estado final | Prob. | Recompensa  |
| :---: | :---: | :---: | :---: | :---: |
| high | search | high | α | r_search |
| high | search | low  | 1-α | r_search |
| low  | search | high | 1-β | -3 |
| low  | search | low  | β | r_search |
| high | wait   | high | 1 | r_wait |
| low  | wait   | low  | 1 | r_wait |
| low  | recharge| high| 1 | 0 |


## 2. Implementação do Algoritmo

### Arquitetura do código

Modularizamos o código em três arquivos: `models.py`, `main.py` e `plot.py`

**models.py** - Onde estão modeladas as classes `Action`, `State`, `Game` e `Player`

- `Action`: identifica ação por `name`
- `State`: identifica estado por `charge_level` e lista as `Action` permitidas por estado.
- `Player`: o robô de reciclagem. Possui os métodos `reset()` para voltar ao estado inicial, `act()` para selecionar a melhor ação a ser tomada naquele momento, `update()` para atualizar os estimadores e `get_policy()`

estimações **V(s)**, escolhe ação com *ε-greedy* a
partir de um backup de um passo que combina recompensas imediatas, transições (α, β) e valores atuais V.\

- `Game`: ambiente com dinâmica (α, β, recompensas). Possui os métodos `transition()` recebe o estado atual e a ação do jogador e retorna o novo estado e a rcompensa da ação.


( V(s) `\leftarrow `{=tex}V(s) + lpha\_{step} ig\[r +
`\gamma `{=tex}V(s') - V(s)ig\] )

-   Observação: o *update* só ocorre se a ação foi *greedy* (seguindo
    Sutton & Barto).

**main.py** - `train(...)`: executa múltiplas épocas, cada uma com
*steps_per_epoch* passos.\
- `main()`: define hiperparâmetros, treina, imprime médias, e plota
resultados via `plot.py`.

**plot.py** - `plot_rewards`: série temporal + histograma de recompensas
por época (com média móvel).\
- `plot_policy_heatmap`: calcula política ε-greedy e plota heatmap por
estado/ação.

**Observação de projeto** - O agente aprende **V(s)** com TD(0), mas
seleciona ações usando um backup *model-based*.\
- Não é Q-learning nem SARSA. É um esquema híbrido (*controle por
one-step lookahead* + avaliação TD).\
- **Vantagem**: simplicidade e uso eficiente do modelo.\
- **Limitação**: sem modelo conhecido, a política não poderia ser
computada dessa forma.




### Loop de Aprendizagem

O treinamento foi dividido em épocas de 1000 passos cada. Ao final de cada época, a recompensa total acumulada foi registrada.

### Definição dos Parâmetros

Os seguintes hiperparâmetros foram utilizados no treinamento:
* **Número de Épocas:** Total de épocas de treinamento.

## 4. Análise dos Resultados

### Curva de Aprendizagem

O gráfico abaixo mostra a recompensa total acumulada ao final de cada época de treinamento.

![Histograma](rewards_plot.png)
![Heatmap](policy_heatmap.png)


* **Pergunta-guia:** O que a política final nos diz? Qual ação o robô aprendeu a tomar no estado `high`? E no estado `low`? Essa política faz sentido intuitivamente? Por quê?

## 5. Conclusão

* **Pergunta-guia:** O algoritmo foi eficaz? Quais foram os maiores desafios na implementação? Que melhorias poderiam ser feitas (ex: variar os parâmetros, testar outro algoritmo)?