# Aprendizado por Reforço para Robô de Reciclagem

## Introdução

Este relatório descreve a implementação de um algoritmo de Aprendizado por Reforço, baseado em *Temporal Difference* (TD), para resolver o problema do Robô de Reciclagem, conforme descrito no Exemplo 3.3 do livro "Reinforcement Learning: An Introduction" de Sutton e Barto. O objetivo do agente é aprender uma política ótima de ações para maximizar a coleta de recompensas em uma tarefa contínua.

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