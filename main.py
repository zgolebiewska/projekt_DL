import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import movie_reviews
import numpy as np

nltk.download('movie_reviews')

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

np.random.shuffle(documents)

def extract_text_and_labels(documents):
    texts = [" ".join(words) for words, _ in documents]
    labels = [category for _, category in documents]
    return texts, labels

texts, labels = extract_text_and_labels(documents)

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
train_features = tfidf_vectorizer.fit_transform(train_texts)
test_features = tfidf_vectorizer.transform(test_texts)

tfidf_model = LogisticRegression()
tfidf_model.fit(train_features, train_labels)

tfidf_predictions = tfidf_model.predict(test_features)
tfidf_accuracy = accuracy_score(test_labels, tfidf_predictions)
tfidf_precision = precision_score(test_labels, tfidf_predictions, average='weighted')
tfidf_recall = recall_score(test_labels, tfidf_predictions, average='weighted')
tfidf_f1 = f1_score(test_labels, tfidf_predictions, average='weighted')

print("TF-IDF Model:")
print("Accuracy:", tfidf_accuracy)
print("Precision:", tfidf_precision)
print("Recall:", tfidf_recall)
print("F1-score:", tfidf_f1)

word2vec_model = Word2Vec([text.split() for text in train_texts], size=100, window=5, min_count=1, workers=4)

def get_doc_vector(text):
    tokens = text.split()
    vecs = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

train_vectors = np.array([get_doc_vector(text) for text in train_texts])
test_vectors = np.array([get_doc_vector(text) for text in test_texts])

word2vec_model = LogisticRegression()
word2vec_model.fit(train_vectors, train_labels)

word2vec_predictions = word2vec_model.predict(test_vectors)
word2vec_accuracy = accuracy_score(test_labels, word2vec_predictions)
word2vec_precision = precision_score(test_labels, word2vec_predictions, average='weighted')
word2vec_recall = recall_score(test_labels, word2vec_predictions, average='weighted')
word2vec_f1 = f1_score(test_labels, word2vec_predictions, average='weighted')

print("\nWord2Vec Model:")
print("Accuracy:", word2vec_accuracy)
print("Precision:", word2vec_precision)
print("Recall:", word2vec_recall)
print("F1-score:", word2vec_f1)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])


def calculate_metrics(model, data_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.float())
            predictions = outputs.argmax(1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return accuracy, precision, recall, f1


input_size = 100
hidden_size = 128
output_size = len(set(train_labels))
learning_rate = 0.001
num_epochs = 10
batch_size = 32

train_dataset = CustomDataset(train_vectors, train_labels)
test_dataset = CustomDataset(test_vectors, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

rnn_model = RNNModel(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    rnn_model.train()
    total_loss = 0
    total_correct = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = rnn_model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {total_correct/len(train_dataset):.4f}')

rnn_accuracy, rnn_precision, rnn_recall, rnn_f1 = calculate_metrics(rnn_model, test_loader)
print("\nRNN Model:")
print("Accuracy:", rnn_accuracy)
print("Precision:", rnn_precision)
print("Recall:", rnn_recall)
print("F1-score:", rnn_f1)
