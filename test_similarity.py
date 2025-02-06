import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import numpy as np
import csv

tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
)

input_text = "Once upon a time, in a quaint little village nestled between rolling hills and lush forests, there existed a village named Green Valley, was home to villagers who lived peaceful and harmonious lives. A clear stream flowed through the village, nourishing the fields and orchards on its banks. Every morning, the villagers would rise early to begin their day’s work. The men worked in the fields, the women wove and cooked at home, and the children studied at the village school.In the village lived an elderly man named Grandpa Li, the village sage. Villagers would seek his advice whenever they faced difficulties. Grandpa Li was not only knowledgeable but also a great storyteller. Every evening, villagers would gather in front of his house to listen to his tales of ancient legends and magical adventures.One day, a stranger arrived in the village. He introduced himself as a traveler named Amin, carrying a mysterious package filled with strange items. The villagers, curious about Amin, gathered around to ask about his origins and experiences. Amin smiled and told them he came from a distant land in the East, having traveled to many places and witnessed many wonders.Amin decided to stay in the village, leading the villagers on explorations of the surrounding forests and hills, uncovering hidden treasures and secrets. Under his guidance, the villagers discovered rare flowers, plants, and animals they had never seen before. Amin also taught them new skills and knowledge, enriching their lives.However, Amin’s arrival was not by chance. He was guided by an ancient prophecy to Green Valley, seeking a legendary mystical power said to be hidden in the village. This power could only be found by those who were pure of heart and brave. Amin believed the villagers of Green Valley were the ones he was looking for.Amin decided to share this secret with the villagers. He gathered them under the big tree in the village and told them about the ancient prophecy. The villagers were both surprised and excited, eager to help Amin find this mystical power. Thus, under Amin’s leadership, they embarked on a journey filled with adventures and challenges.They traversed dense forests, climbed steep mountains, and overcame numerous obstacles. Eventually, they found the mystical power in a hidden cave. This power not only made the village more prosperous but also purified and elevated the villagers’ spirits.From then on, Green Valley became even more beautiful and harmonious, and the villagers grew more united and loving. Amin continued his journey, seeking more miracles and adventures. The story of Green Valley spread far and wide, told by Grandpa Li to all who would listen."

tokenized_input = tokenizer(input_text, return_tensors="pt", truncation=True)
with torch.no_grad():
    outputs = model(**tokenized_input,  output_attentions=True)

# Access attention weights
attentions = outputs.attentions  # List of attention tensors for each layer

# Ensure that the model actually output attentions
if attentions is None:
    raise ValueError("The model did not output attentions. "
                     "Please ensure that `output_attentions=True` is set when initializing the model.")

# Function to calculate attention weight similarity (e.g., using cosine similarity)
def calculate_attention_similarity(attn_weights1, attn_weights2):
    # attn_weights: (batch_size, num_heads, seq_len, seq_len)
    # Flatten the attention weights for each head
    attn_weights1 = attn_weights1.view(-1, attn_weights1.size(-2), attn_weights1.size(-1))
    attn_weights2 = attn_weights2.view(-1, attn_weights2.size(-2), attn_weights1.size(-1))

    # Calculate cosine similarity between attention weights of corresponding heads
    similarity_matrix = torch.zeros(attn_weights1.size(-2), attn_weights2.size(-2))  # (num_heads1, num_heads2)
    for i in range(attn_weights1.size(1)):
        for j in range(attn_weights2.size(1)):
            similarity_matrix[i, j] = F.cosine_similarity(attn_weights1[5, :, i], attn_weights2[5, :, j], dim=-1).mean()

    return similarity_matrix

# Base layer
base_layer = 9
base_attn_weights = attentions[base_layer]  # (batch_size, num_heads, seq_len, seq_len)

# Comparison layers
comparison_layers = [9, 5, 10, 15, 20, 25, 30]
similarity_matrices_attn = {}

# Calculate attention weight similarities
for layer in comparison_layers:
    comparison_attn_weights = attentions[layer]
    similarity_matrices_attn[layer] = calculate_attention_similarity(base_attn_weights, comparison_attn_weights)

# Plotting
num_plots = len(comparison_layers)
plt.figure(figsize=(8 * num_plots, 8))

# Heatmaps for attention weight similarity
for i, layer in enumerate(comparison_layers):
    plt.subplot(1, num_plots, i + 1)
    plt.imshow(similarity_matrices_attn[layer].cpu(), cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.xlabel(f"Heads (Layer {layer})")
    plt.ylabel(f"Heads (Layer {base_layer})")
    plt.title(f"Attention Weight Similarity (Layer {base_layer} vs. Layer {layer})")

plt.tight_layout()
plt.savefig("attention_weight_similarity_heatmaps.png")
print("Attention weight similarity heatmaps saved to attention_weight_similarity_heatmaps.png")