import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('bert_recommendation_model.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract training and test data
train_embeddings = data['train_embeddings']
train_labels = data['train_labels']
test_embeddings = data['test_embeddings']
test_issues = data['test_issues']
test_labels = data['test_labels']
tokenizer = data['tokenizer']

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map labels to range [0, num_classes-1] if necessary
unique_labels = sorted(set(train_labels))
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
train_labels_mapped = train_labels.map(label_mapping)
test_labels_mapped = test_labels.map(label_mapping)

# Compute class weights
#unique_labels = np.unique(train_labels_mapped.to_numpy())  # Ensure this is a NumPy array
#y_labels = train_labels_mapped.to_numpy()  # Convert to NumPy array if needed
#class_weights = compute_class_weight('balanced', classes=unique_labels, y=y_labels)
#class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

#prepare tensors
train_labels_tensor = torch.tensor(train_labels_mapped.to_numpy(), dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels_mapped.to_numpy(), dtype=torch.long)
num_classes = len(unique_labels)
train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32)
test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32)

# Create DataLoader for training and testing datasets
batch_size = 32

train_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_embeddings_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Feedforward Neural Network model
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Model parameters
input_size = train_embeddings_tensor.shape[1]
hidden_size = 256
num_classes = len(unique_labels)
dropout_rate = 0.3
learning_rate = 0.0005
num_epochs = 20

# Initialize the model, loss function, and optimizer
model = FNN(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # Use class weights here if needed
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, axis=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Generate classification report
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(all_labels, all_preds,zero_division=0))

torch.save(model.state_dict(), "fnn_classifier.pth")
print('model saved to fnn_classifier.pth')

# Confusion matrix
#cm = confusion_matrix(all_labels, all_preds)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#disp.plot()
#plt.show()