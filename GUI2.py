import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QLineEdit, QTextBrowser, QWidget, QComboBox, QTableWidget, QTableWidgetItem, QFormLayout
from PyQt5.QtCore import Qt
import pandas as pd
import torch
import pickle
from transformers import BertModel
import torch.nn.functional as F
import openpyxl

# Load the model and other components
def load_model_and_data():
    model_path = "fnn_classifier.pth"
    input_size = 768
    hidden_size = 256
    num_classes = 313

    class FNN(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(FNN, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.bn1 = torch.nn.BatchNorm1d(hidden_size)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.5)
            self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNN(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    with open("bert_recommendation_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    tokenizer = model_data["tokenizer"]
    label_to_resolution = model_data["label_to_resolution"]
    return model, tokenizer, label_to_resolution, device

# Get embedding from BERT
def get_embedding(text, tokenizer, bert_model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).to(device)
    return embedding

# Perform recommendation
def recommend_resolution(new_issue, model, tokenizer, label_to_resolution, device):
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    embedding = get_embedding(new_issue, tokenizer, bert_model, device)
    with torch.no_grad():
        outputs = model(embedding)
        probabilities = F.softmax(outputs, dim=1)
        top5_probs, top5_classes = torch.topk(probabilities, 5)

    recommendations = [
        (label_to_resolution[class_idx.item()], prob.item())
        for class_idx, prob in zip(top5_classes[0], top5_probs[0])
    ]
    return recommendations

# PyQt5 GUI Class
class RecommenderUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recommender System")
        self.setGeometry(100, 100, 800, 800)

        self.model, self.tokenizer, self.label_to_resolution, self.device = load_model_and_data()

        self.data = pd.read_excel('Cement Mill 3 Summary Shift Report 2023.xlsx', sheet_name='C.Mill 3 (FYP)')
        self.data.dropna(axis=1, how='all', inplace=True)  # Remove columns with all NaN values

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Enter Issue Description:")
        layout.addWidget(self.label)

        self.input_box = QLineEdit()
        self.input_box.returnPressed.connect(self.get_recommendations)
        layout.addWidget(self.input_box)

        self.recommend_button = QPushButton("Get Recommendations")
        self.recommend_button.clicked.connect(self.get_recommendations)
        layout.addWidget(self.recommend_button)

        self.result_display = QTextBrowser()
        layout.addWidget(self.result_display)

        self.search_label = QLabel("Search Historical Data:")
        layout.addWidget(self.search_label)

        self.filter_dropdown = QComboBox()
        self.filter_dropdown.addItems(["Date", "Equipment #", "Issue"])
        layout.addWidget(self.filter_dropdown)

        self.search_box = QLineEdit()
        layout.addWidget(self.search_box)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_data)
        layout.addWidget(self.search_button)

        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.append_label = QLabel("Append New Entry:")
        layout.addWidget(self.append_label)

        self.append_form = QFormLayout()

        self.date_input = QLineEdit()
        self.append_form.addRow("Date:", self.date_input)

        self.equipment_input = QLineEdit()
        self.append_form.addRow("Equipment #:", self.equipment_input)

        self.issue_input = QLineEdit()
        self.append_form.addRow("Issue:", self.issue_input)

        self.resolution_input = QLineEdit()
        self.append_form.addRow("Resolution:", self.resolution_input)

        self.append_button = QPushButton("Append to Excel")
        self.append_button.clicked.connect(self.append_entry)
        layout.addLayout(self.append_form)
        layout.addWidget(self.append_button)

        self.append_status = QTextBrowser()
        layout.addWidget(self.append_status)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def append_entry(self):
        new_entry = {
            'Date': self.date_input.text().strip(),
            'Equipment #': self.equipment_input.text().strip(),
            'Issues': self.issue_input.text().strip(),
            'Resolution': self.resolution_input.text().strip()
        }
        new_row = pd.DataFrame([new_entry])

        with pd.ExcelWriter('Cement Mill 3 Summary Shift Report 2023.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            existing_data = pd.read_excel(writer, sheet_name='C.Mill 3 (FYP)')
            updated_data = pd.concat([existing_data, new_row], ignore_index=True)
            updated_data.to_excel(writer, sheet_name='C.Mill 3 (FYP)', index=False)

        self.append_status.setText("New entry appended successfully!")

    def get_recommendations(self):
        issue_description = self.input_box.text().strip()
        if not issue_description:
            self.result_display.setText("Please enter a valid issue description.")
            return

        recommendations = recommend_resolution(
            issue_description, self.model, self.tokenizer, self.label_to_resolution, self.device
        )

        result_text = "Top Recommendations:\n"
        for idx, (resolution, confidence) in enumerate(recommendations, 1):
            result_text += f"{idx}. {resolution} (Confidence: {confidence:.2f})\n"

        self.result_display.setText(result_text)

    def search_data(self):
        query = self.search_box.text().strip()
        filter_by = self.filter_dropdown.currentText()

        if filter_by == "Date":
            results = self.data[self.data['Date'].astype(str).str.contains(query, case=False, na=False)]
        elif filter_by == "Equipment #":
            results = self.data[self.data['Equipment #'].astype(str).str.contains(query, case=False, na=False)]
        else:
            results = self.data[self.data['Issues'].astype(str).str.contains(query, case=False, na=False)]

        self.table.setRowCount(len(results))
        self.table.setColumnCount(len(results.columns))
        self.table.setHorizontalHeaderLabels(results.columns)

        for i, (index, row) in enumerate(results.iterrows()):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecommenderUI()
    window.show()
    sys.exit(app.exec_())
