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
    # Enable use_cache to retrieve past_key_values (includes value states)
    outputs = model(**tokenized_input, output_attentions=True, use_cache=True)

# Access attention weights and past key-value states
attentions = outputs.attentions
past_key_values = outputs.past_key_values

def calculate_attention_output_similarity(base_attn_weights, base_values, comp_attn_weights, comp_values):
    # Remove batch dimension (assuming batch_size=1)
    base_attn_weights = base_attn_weights.squeeze(0)  # (num_heads, seq_len, seq_len)
    base_values = base_values.squeeze(0)              # (num_heads, seq_len, head_dim)
    comp_attn_weights = comp_attn_weights.squeeze(0)
    comp_values = comp_values.squeeze(0)
    
    # Compute attention outputs: (num_heads, seq_len, head_dim)
    base_output = torch.matmul(base_attn_weights, base_values)
    comp_output = torch.matmul(comp_attn_weights, comp_values)
    
    # Flatten to (num_heads, seq_len * head_dim)
    base_output = base_output.view(base_output.size(0), -1)
    comp_output = comp_output.view(comp_output.size(0), -1)
    
    # Calculate cosine similarity between each head pair
    num_base_heads = base_output.size(0)
    num_comp_heads = comp_output.size(0)
    similarity_matrix = torch.zeros(num_base_heads, num_comp_heads)
    for i in range(num_base_heads):
        for j in range(num_comp_heads):
            similarity_matrix[i, j] = F.cosine_similarity(
                base_output[i].unsqueeze(0), 
                comp_output[j].unsqueeze(0), 
                dim=1
            ).item()
    
    return similarity_matrix

# Base layer (adjust index if needed)
base_layer = 19
base_attn_weights = attentions[base_layer]
base_values = past_key_values[base_layer][1]  # past_key_values[layer][1] is the value tensor

# Comparison layers
comparison_layers = [19, 5, 10, 15, 20, 25, 30]
similarity_matrices = {}

for layer in comparison_layers:
    comp_attn_weights = attentions[layer]
    comp_values = past_key_values[layer][1]  # Get value state for the comparison layer
    similarity_matrices[layer] = calculate_attention_output_similarity(
        base_attn_weights, base_values, comp_attn_weights, comp_values
    )

# Generate heatmaps
num_plots = len(comparison_layers)
plt.figure(figsize=(8 * num_plots, 8))

for i, layer in enumerate(comparison_layers):
    plt.subplot(1, num_plots, i + 1)
    plt.imshow(similarity_matrices[layer].cpu(), cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.xlabel(f"Heads (Layer {layer})")
    plt.ylabel(f"Heads (Layer {base_layer})")
    plt.title(f"Attention Output Similarity\nLayer {base_layer} vs. {layer}")

plt.tight_layout()
plt.savefig("attention_output_similarity_heatmaps.png")
print("Heatmaps saved to attention_output_similarity_heatmaps.png")