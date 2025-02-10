import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import numpy as np
import csv

tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/",
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/",
)

input_text = "Once upon a time, in a quaint little village nestled between rolling hills and lush forests, there existed a village named Green Valley, was home to villagers who lived peaceful and harmonious lives. A clear stream flowed through the village, nourishing the fields and orchards on its banks. Every morning, the villagers would rise early to begin their day’s work. The men worked in the fields, the women wove and cooked at home, and the children studied at the village school.In the village lived an elderly man named Grandpa Li, the village sage. Villagers would seek his advice whenever they faced difficulties. Grandpa Li was not only knowledgeable but also a great storyteller. Every evening, villagers would gather in front of his house to listen to his tales of ancient legends and magical adventures.One day, a stranger arrived in the village. He introduced himself as a traveler named Amin, carrying a mysterious package filled with strange items. The villagers, curious about Amin, gathered around to ask about his origins and experiences. Amin smiled and told them he came from a distant land in the East, having traveled to many places and witnessed many wonders.Amin decided to stay in the village, leading the villagers on explorations of the surrounding forests and hills, uncovering hidden treasures and secrets. Under his guidance, the villagers discovered rare flowers, plants, and animals they had never seen before. Amin also taught them new skills and knowledge, enriching their lives.However, Amin’s arrival was not by chance. He was guided by an ancient prophecy to Green Valley, seeking a legendary mystical power said to be hidden in the village. This power could only be found by those who were pure of heart and brave. Amin believed the villagers of Green Valley were the ones he was looking for.Amin decided to share this secret with the villagers. He gathered them under the big tree in the village and told them about the ancient prophecy. The villagers were both surprised and excited, eager to help Amin find this mystical power. Thus, under Amin’s leadership, they embarked on a journey filled with adventures and challenges.They traversed dense forests, climbed steep mountains, and overcame numerous obstacles. Eventually, they found the mystical power in a hidden cave. This power not only made the village more prosperous but also purified and elevated the villagers’ spirits.From then on, Green Valley became even more beautiful and harmonious, and the villagers grew more united and loving. Amin continued his journey, seeking more miracles and adventures. The story of Green Valley spread far and wide, told by Grandpa Li to all who would listen."

# Tokenize input
tokenized_input = tokenizer(input_text, return_tensors="pt", truncation=True)

# Forward pass with attention and past_key_values
with torch.no_grad():
    outputs = model(**tokenized_input, output_attentions=True, use_cache=True)

# Extract attentions and past_key_values
attentions = outputs.attentions  # List of attention tensors for each layer
past_key_values = outputs.past_key_values  # List of key-value states for each layer

# Get total number of layers
num_layers = model.config.num_hidden_layers
# Ensure that the model actually output attentions
if attentions is None:
    raise ValueError("The model did not output attentions. "
                     "Please ensure that `output_attentions=True` is set when initializing the model.")

# Function to calculate attention weight similarity (e.g., using cosine similarity)
def calculate_attention_similarity(v1, v2, i, j):
    with torch.no_grad():  # Ensure no gradients are tracked
        print(i)
        seq_len = v1.shape[-2]
        proj_o1 = model.model.layers[i].self_attn.o_proj
        proj_o2 = model.model.layers[j].self_attn.o_proj
        v1 = v1.transpose(1, 2).contiguous()
        v1 = proj_o1(v1.reshape(1, seq_len, 128*32))
        v2 = v2.transpose(1, 2).contiguous()
        v2 = proj_o2(v2.reshape(1, seq_len, 128*32))
        v1 = v1.view(1, seq_len, 32, 128).transpose(1, 2)
        v2 = v2.view(1, seq_len, 32, 128).transpose(1, 2)
        
        v1 = v1.squeeze(0)  # Shape: [32, n, 128]
        v2 = v2.squeeze(0)  # Shape: [32, n, 128]

        similarity_matrix = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            print(i)
            for j in range(seq_len):
                similarity_matrix[i, j] = F.cosine_similarity(v1[:, i, :], v2[:, j, :], dim=-1).mean()
        print(similarity_matrix.mean())
    return similarity_matrix.detach().cpu()  # Detach and move to CPU



    # 0.0817

# Base layer
base_layer = 9
base_attn_weights = past_key_values[base_layer][1]  # (batch_size, num_heads, seq_len, seq_len)

# Comparison layers
comparison_layers = [5]
similarity_matrices_attn = {}

# Calculate attention weight similarities
for layer in comparison_layers:
    comparison_attn_weights = past_key_values[layer][1]
    similarity_matrices_attn[layer] = calculate_attention_similarity(comparison_attn_weights, comparison_attn_weights, layer, layer)

# Plotting
num_plots = len(comparison_layers)
plt.figure(figsize=(8 * num_plots, 8))

# Heatmaps for attention weight similarity
for i, layer in enumerate(comparison_layers):
    plt.subplot(1, num_plots, i + 1)
    plt.imshow(similarity_matrices_attn[layer].numpy(), cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.xlabel(f"Heads (Layer {layer})")
    plt.ylabel(f"Heads (Layer {base_layer})")
    plt.title(f"Attention Weight Similarity (Layer {base_layer} vs. Layer {layer})")

plt.tight_layout()
plt.savefig("proj_value_similarity_heatmaps.png")
print("Attention weight similarity heatmaps saved to attention_weight_similarity_heatmaps.png")