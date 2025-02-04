import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import numpy as np
import csv

tokenizer = AutoTokenizer.from_pretrained(
     "/root/autodl-tmp/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",  # Use the appropriate model name
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
     "/root/autodl-tmp/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",  # Use the appropriate model name
    torch_dtype=torch.bfloat16, # Necessary for Llama-3-8B-Instruct, since it is trained using bfloat16
    device_map="auto"
)

input_text = "Once upon a time, in a quaint little village nestled between rolling hills and lush forests, there existed a village named Green Valley, was home to villagers who lived peaceful and harmonious lives. A clear stream flowed through the village, nourishing the fields and orchards on its banks. Every morning, the villagers would rise early to begin their day’s work. The men worked in the fields, the women wove and cooked at home, and the children studied at the village school.In the village lived an elderly man named Grandpa Li, the village sage. Villagers would seek his advice whenever they faced difficulties. Grandpa Li was not only knowledgeable but also a great storyteller. Every evening, villagers would gather in front of his house to listen to his tales of ancient legends and magical adventures.One day, a stranger arrived in the village. He introduced himself as a traveler named Amin, carrying a mysterious package filled with strange items. The villagers, curious about Amin, gathered around to ask about his origins and experiences. Amin smiled and told them he came from a distant land in the East, having traveled to many places and witnessed many wonders.Amin decided to stay in the village, leading the villagers on explorations of the surrounding forests and hills, uncovering hidden treasures and secrets. Under his guidance, the villagers discovered rare flowers, plants, and animals they had never seen before. Amin also taught them new skills and knowledge, enriching their lives.However, Amin’s arrival was not by chance. He was guided by an ancient prophecy to Green Valley, seeking a legendary mystical power said to be hidden in the village. This power could only be found by those who were pure of heart and brave. Amin believed the villagers of Green Valley were the ones he was looking for.Amin decided to share this secret with the villagers. He gathered them under the big tree in the village and told them about the ancient prophecy. The villagers were both surprised and excited, eager to help Amin find this mystical power. Thus, under Amin’s leadership, they embarked on a journey filled with adventures and challenges.They traversed dense forests, climbed steep mountains, and overcame numerous obstacles. Eventually, they found the mystical power in a hidden cave. This power not only made the village more prosperous but also purified and elevated the villagers’ spirits.From then on, Green Valley became even more beautiful and harmonious, and the villagers grew more united and loving. Amin continued his journey, seeking more miracles and adventures. The story of Green Valley spread far and wide, told by Grandpa Li to all who would listen."

tokenized_input = tokenizer(input_text, return_tensors="pt", truncation=True).to("cuda") # Move input tensors to GPU
with torch.no_grad():
    outputs = model(**tokenized_input, output_hidden_states=True, use_cache=True) # Modified to output past_key_values

# Access past_key_values
past_key_values = outputs.past_key_values

# Ensure that the model actually output past_key_values
if past_key_values is None:
    raise ValueError("The model did not output past_key_values. "
                     "Please ensure that `use_cache=True` is set when initializing the model.")

# Function to calculate key state similarity (e.g., using cosine similarity)
def calculate_key_state_similarity(key_states1, key_states2):
    # key_states: (batch_size, num_heads, seq_len, head_dim)
    # Flatten the key states for each head
    key_states1 = key_states1[0].view(-1, key_states1.size(-2), key_states1.size(-1)) # shape: (batch_size, num_heads, seq_len * head_dim)
    print(key_states1.shape)
    key_states2 = key_states2[0].view(-1, key_states1.size(-2), key_states1.size(-1))

    # Calculate cosine similarity between key states of corresponding heads
    similarity_matrix = torch.zeros(key_states1.size(-2), key_states2.size(-2))  # (num_heads1, num_heads2)
    for i in range(key_states1.size(1)):
        for j in range(key_states2.size(1)):
            similarity_matrix[i, j] = F.cosine_similarity(key_states1[0, i, :], key_states2[0, j, :], dim=-1) # Removed .mean() since we are comparing single vectors

    return similarity_matrix

# Base layer
base_layer = 4
base_key_states = past_key_values[base_layer][1] # past_key_values structure: num_layers x (key_tensor, value_tensor), key/value_tensor shape: (batch_size, num_heads, seq_len, head_dim)

# Comparison layers
comparison_layers = [4, 5, 10, 15, 20, 25, 30]
similarity_matrices_key = {}

# Calculate key state similarities
for layer in comparison_layers:
    comparison_key_states = past_key_values[layer][1]
    similarity_matrices_key[layer] = calculate_key_state_similarity(base_key_states, comparison_key_states)

# Plotting
num_plots = len(comparison_layers)
plt.figure(figsize=(8 * num_plots, 8))

# Heatmaps for key state similarity
for i, layer in enumerate(comparison_layers):
    plt.subplot(1, num_plots, i + 1)
    plt.imshow(similarity_matrices_key[layer].cpu(), cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.xlabel(f"Heads (Layer {layer})")
    plt.ylabel(f"Heads (Layer {base_layer})")
    plt.title(f"Key State Similarity (Layer {base_layer} vs. Layer {layer})")

plt.tight_layout()
plt.savefig("key_state_similarity_heatmaps.png")
print("Key state similarity heatmaps saved to key_state_similarity_heatmaps.png")