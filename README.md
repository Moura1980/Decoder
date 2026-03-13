Requisitos:
pip install numpy

Como executar:
python laboratorio3.py

O que está implementado:
Tarefa 1 — Máscara Causal
Cria uma matriz triangular que impede o modelo de olhar para palavras futuras durante o treinamento. Posições futuras recebem -inf, que o Softmax converte em 0.0.

Tarefa 2 — Cross-Attention
Ponte entre o Encoder e o Decoder. O Decoder usa o que já gerou para perguntar (Query), e a saída do Encoder responde com contexto (Keys e Values). Sem máscara, pois o Encoder já está completo.

Tarefa 3 — Loop Auto-Regressivo
Simula a geração token a token. A cada passo, o modelo escolhe a palavra com maior probabilidade via argmax e a adiciona à sequência, até gerar <EOS>.
