import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import torch
import time

start_time = time.time()

# Step 1: Read and Clean Dataset
df = pd.read_excel('Cement Mill 3 Summary Shift Report 2023.xlsx', sheet_name='C.Mill 3 (visualize)')
df = df.dropna(how='all', axis=1)  # Remove columns with all NaN
df_cleaned = df.dropna(how='any', axis=0)  # Remove rows with any NaN

# Extract Issues and Resolutions
issues = df_cleaned['Issues'].str.lower().str.strip()
resolutions = df_cleaned['Resolution'].str.lower().str.strip()

# Encode Resolutions to Numeric Labels
resolution_classes = resolutions.unique()
resolution_to_label = {resolution: idx for idx, resolution in enumerate(resolution_classes)}
label_to_resolution = {idx: resolution for resolution, idx in resolution_to_label.items()}
labels = resolutions.map(resolution_to_label)

# Step 2: Train-Test Split
train_issues, test_issues, train_labels, test_labels = train_test_split(
    issues, labels, test_size=0.2, random_state=42
)

# Step 3: Load Pre-trained BERT Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to Generate Embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()  # Move back to CPU for saving

# Generate Embeddings for Training Data and Testing Data
train_embeddings = [get_embedding(issue) for issue in train_issues]
train_embeddings = np.vstack(train_embeddings)  # Combine into a 2D array
test_embeddings = [get_embedding(issue) for issue in test_issues]
test_embeddings = np.vstack(test_embeddings)  # Combine into a 2D array

# Save the Embeddings, Labels, and Model
with open('bert_recommendation_model.pkl', 'wb') as f:
    pickle.dump({
        'train_embeddings': train_embeddings,
        'train_issues': train_issues,
        'train_labels': train_labels,
        'tokenizer': tokenizer,
        'test_embeddings' : test_embeddings,
        'test_issues': test_issues,
        'test_labels': test_labels,
        'label_to_resolution': label_to_resolution,
        'resolution_to_label': resolution_to_label,
    }, f)

print("Model and data saved successfully!")

end_time = time.time()
time_taken = end_time - start_time
minutes, seconds = divmod(time_taken, 60)
print('model saved to bert_recommendation_model.pkl')
print(f"Time taken to build the model: {int(minutes)} minutes and {seconds:.2f} seconds")
