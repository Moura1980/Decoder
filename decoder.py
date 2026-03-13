import numpy as np


#Máscara Causal (Look-Ahead Mask)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def create_causal_mask(seq_len):
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask

seq_len = 5
mask = create_causal_mask(seq_len)

print("Máscara Causal")
print("Máscara M:")
print(mask)

Q = np.random.randn(seq_len, seq_len)
K = np.random.randn(seq_len, seq_len)

scores = Q @ K.T + mask
resultado_softmax = softmax(scores)

print("\nResultado após Softmax:")
print(np.round(resultado_softmax, 4))


#Cross-Attention (Ponte Encoder-Decoder)
def cross_attention(encoder_out, decoder_state):
    batch_size, seq_len_enc, d_model = encoder_out.shape
    _, seq_len_dec, _ = decoder_state.shape

    Wq = np.random.randn(d_model, d_model) * 0.01
    Wk = np.random.randn(d_model, d_model) * 0.01
    Wv = np.random.randn(d_model, d_model) * 0.01

    Q = decoder_state @ Wq
    K = encoder_out @ Wk
    V = encoder_out @ Wv

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)
    attn_weights = softmax(scores)
    output = attn_weights @ V

    return output, attn_weights

batch_size = 1
seq_len_frances = 10
seq_len_ingles = 4
d_model = 512

encoder_output = np.random.randn(batch_size, seq_len_frances, d_model)
decoder_state = np.random.randn(batch_size, seq_len_ingles, d_model)

cross_out, attn_weights = cross_attention(encoder_output, decoder_state)

print("\nCross-Attention")
print(f"Shape encoder_output: {encoder_output.shape}")
print(f"Shape decoder_state:  {decoder_state.shape}")
print(f"Shape saída Cross-Attention: {cross_out.shape}")
print(f"Shape pesos de atenção: {attn_weights.shape}")
print(f"Soma dos pesos (deve ser ~1.0): {np.round(attn_weights[0][0].sum(), 4)}")


#Loop de Inferência Auto-Regressivo
VOCAB_SIZE = 10000
EOS_TOKEN_ID = 1

vocabulario = {i: f"palavra_{i}" for i in range(VOCAB_SIZE)}
vocabulario[0] = "<START>"
vocabulario[EOS_TOKEN_ID] = "<EOS>"
vocabulario[2] = "O"
vocabulario[3] = "rato"
vocabulario[4] = "roeu"
vocabulario[5] = "a"
vocabulario[6] = "roupa"

def generate_next_token(current_sequence, encoder_out):
    logits = np.random.randn(VOCAB_SIZE)
    probs = softmax(logits)
    return probs

print("\nLoop de Inferência Auto-Regressivo")

current_sequence = ["<START>"]
encoder_out = np.random.randn(1, seq_len_frances, d_model)

max_tokens = 20
step = 0

while step < max_tokens:
    probs = generate_next_token(current_sequence, encoder_out)
    next_token_id = int(np.argmax(probs))
    next_word = vocabulario[next_token_id]

    current_sequence.append(next_word)
    print(f"Passo {step + 1}: token gerado = '{next_word}' (id={next_token_id})")

    if next_word == "<EOS>":
        print("\nToken <EOS> detectado! Geração encerrada.")
        break

    step += 1

print("\nFrase final gerada:")
print(" ".join(current_sequence))