import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/",
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/",
).to(device)  # Move model to GPU

input_text = "Once upon a time, in a quaint little village nestled between rolling hills and lush forests, there existed a village named Green Valley, was home to villagers who lived peaceful and harmonious lives. A clear stream flowed through the village, nourishing the fields and orchards on its banks. Every morning, the villagers would rise early to begin their day’s work. The men worked in the fields, the women wove and cooked at home, and the children studied at the village school.In the village lived an elderly man named Grandpa Li, the village sage. Villagers would seek his advice whenever they faced difficulties. Grandpa Li was not only knowledgeable but also a great storyteller. Every evening, villagers would gather in front of his house to listen to his tales of ancient legends and magical adventures.One day, a stranger arrived in the village. He introduced himself as a traveler named Amin, carrying a mysterious package filled with strange items. The villagers, curious about Amin, gathered around to ask about his origins and experiences. Amin smiled and told them he came from a distant land in the East, having traveled to many places and witnessed many wonders.Amin decided to stay in the village, leading the villagers on explorations of the surrounding forests and hills, uncovering hidden treasures and secrets. Under his guidance, the villagers discovered rare flowers, plants, and animals they had never seen before. Amin also taught them new skills and knowledge, enriching their lives.However, Amin’s arrival was not by chance. He was guided by an ancient prophecy to Green Valley, seeking a legendary mystical power said to be hidden in the village. This power could only be found by those who were pure of heart and brave. Amin believed the villagers of Green Valley were the ones he was looking for.Amin decided to share this secret with the villagers. He gathered them under the big tree in the village and told them about the ancient prophecy. The villagers were both surprised and excited, eager to help Amin find this mystical power. Thus, under Amin’s leadership, they embarked on a journey filled with adventures and challenges.They traversed dense forests, climbed steep mountains, and overcame numerous obstacles. Eventually, they found the mystical power in a hidden cave. This power not only made the village more prosperous but also purified and elevated the villagers’ spirits.From then on, Green Valley became even more beautiful and harmonious, and the villagers grew more united and loving. Amin continued his journey, seeking more miracles and adventures. The story of Green Valley spread far and wide, told by Grandpa Li to all who would listen."

# Tokenize input and move to GPU
tokenized_input = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

# Forward pass to get attention weights
with torch.no_grad():
    outputs = model(**tokenized_input, output_attentions=True, use_cache=True)

attentions = outputs.attentions  # List of (batch_size, num_heads, seq_len, seq_len) tensors

# Calculate average attention per layer (on GPU)
layer_attentions = [attn[0][0] for attn in attentions]  # List of (seq_len, seq_len) tensors
num_layers = len(layer_attentions)
seq_len = layer_attentions[0].size(0)

# Calculate layer similarity matrix using cosine similarity (on GPU)
layer_sim_matrix = torch.zeros((num_layers, num_layers), device=device)
for i in range(num_layers):
    for j in range(num_layers):
        vec_i = layer_attentions[i].flatten()
        vec_j = layer_attentions[j].flatten()
        cosine_sim = torch.dot(vec_i, vec_j) / (torch.norm(vec_i) * torch.norm(vec_j))
        layer_sim_matrix[i, j] = cosine_sim

# Find most similar layer pair (on GPU)
max_sim = -1
best_pair = (0, 1)
for i in range(num_layers):
    for j in range(i+1, num_layers):
        if layer_sim_matrix[i, j] > max_sim:
            max_sim = layer_sim_matrix[i, j]
            best_pair = (i, j)

# Calculate token-level similarity for best layer pair (on GPU)
l1, l2 = 7,11
token_sims = []
attn_weight_sum_l1 = []
attn_weight_sum_l2 = []
for i in range(seq_len):
    row_i_l1 = layer_attentions[l1][i, :].flatten()
    row_i_l2 = layer_attentions[l2][i, :].flatten()
    sim = torch.dot(row_i_l1, row_i_l2) / (torch.norm(row_i_l1) * torch.norm(row_i_l2))
    print(sim)
    token_sims.append(sim)
    attn_weight_sum_l1.append(layer_attentions[l1][i, :].sum(dim=-1).item())
    attn_weight_sum_l2.append(layer_attentions[l2][i, :].sum(dim=-1).item())

token_sims_matrix = torch.stack(token_sims).reshape(1, -1)  # Shape (1, seq_len)
attn_weight_sum_l1_matrix = torch.stack(attn_weight_sum_l1).reshape(1, -1)  # Shape (1, seq_len)
attn_weight_sum_l2_matrix = torch.stack(attn_weight_sum_l2).reshape(1, -1)  # Shape (1, seq_len)

# Move data to CPU for plotting
layer_sim_matrix = layer_sim_matrix.cpu().numpy()
token_sims_matrix = token_sims_matrix.cpu().numpy()
attn_weight_sum_l2_matrix = attn_weight_sum_l2_matrix.cpu().numpy()
attn_weight_sum_l1_matrix = attn_weight_sum_l1_matrix.cpu().numpy()

# Create combined plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

# Layer similarity heatmap
im1 = ax1.imshow(layer_sim_matrix, cmap='viridis', vmin=0, vmax=1)
ax1.set_title('Layer Similarity Matrix', fontsize=14)
ax1.set_xlabel('Layer Index', fontsize=12)
ax1.set_ylabel('Layer Index', fontsize=12)
ax1.set_xticks(range(num_layers))
ax1.set_yticks(range(num_layers))
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Token similarity heatmap
im2 = ax2.imshow(token_sims_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
ax2.set_title(f'Token-level Similarity (Layers {l1} & {l2})', fontsize=14)
ax2.set_xlabel('Token Position', fontsize=12)
ax2.set_yticks([])
ax2.set_xticks(range(0, seq_len, max(1, seq_len//10)))
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# Attention weight sum heatmap
im3 = ax3.imshow(attn_weight_sum_l1_matrix, cmap='viridis', aspect='auto')
ax3.set_title(f'Attention Weight Sum (Layer {l1})', fontsize=14)
ax3.set_xlabel('Token Position', fontsize=12)
ax3.set_yticks([])
ax3.set_xticks(range(0, seq_len, max(1, seq_len//10)))

im4 = ax4.imshow(attn_weight_sum_l2_matrix, cmap='viridis', aspect='auto')
ax4.set_title(f'Attention Weight Sum (Layer {l2})', fontsize=14)
ax4.set_xlabel('Token Position', fontsize=12)
ax4.set_yticks([])
ax4.set_xticks(range(0, seq_len, max(1, seq_len//10)))


plt.tight_layout()
plt.savefig('combined_similarity_heatmap.jpg', dpi=300, bbox_inches='tight')
plt.close()