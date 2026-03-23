from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt

model = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

inputs = tokenizer("I am driving to Paris", return_tensors="pt")
outputs = model(**inputs)

attentions = outputs.attentions

print("\n-- Struktur ---------------------")
# 12 Layer (Transformer-Blöcke); 1 attention matrix pro Layer
print(f"attentions length (=layers count): {len(attentions)}")
# Dims: (batch_size, num_heads, seq_len, seq_len): torch.Size([1, 12, 7, 7]
print(f"Layer 0 shape: {attentions[0].shape}")
print(f"Layer 0, Batch 0 shape: {attentions[0][0].shape}")
print(f"Layer 0, Batch 0, Head 0 shape: {attentions[0][0][0].shape}")
print(f"Layer 0, Batch 0, Head 0, Token 1 Attention (auf alle 7 Tokens): {attentions[0][0][0][1].tolist()}")

print("\n-- Tokens -----------------")
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for i, token in enumerate(tokens):
    print(i, token)

driving = 3 # driving
paris = 5 # Paris
print(f"Attention von '{tokens[driving]}' → '{tokens[paris]}': {attentions[0][0][0][driving][paris]}")

print(f"\nAttention-Verteilung für Token im Layer 1 '{tokens[driving]}':")
for i, token in enumerate(tokens):
    score = attentions[0][0][0][driving][i]
    print(f"→ '{token}': {score:.3f}")

print(f"\nAttention-Verteilung für Token im letzten Layer '{tokens[driving]}':")
for i, token in enumerate(tokens):
    score = attentions[-1][0][0][driving][i]
    print(f"→ '{token}': {score:.3f}")


print(f"\nAttention-Verteilung für Token im Layer 3 in allen Heads '{tokens[driving]}':")
layer = attentions[2][0]  # Layer 3, batch 0
for head in range(layer.shape[0]):
    distribution = layer[head][driving]  # Attention von Token driving
    max = distribution.argmax().item()
    print(f"Head {head + 1}: '{tokens[driving]}' max → '{tokens[max]}' ({distribution[max]:.3f})")
    for paris, score in enumerate(distribution):
        print(f"    → {tokens[paris]}: {score:.3f}")


layer = layer[0].detach().numpy()

plt.imshow(layer, cmap='viridis')
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)

plt.title("Attention (Layer 3, head 1)")
plt.colorbar()
plt.show()