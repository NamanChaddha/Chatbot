import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pickle

# ================= CONFIGURATION =================
SAMPLES_PER_CLASS = 15000
MAX_FEATURES = 9000
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128

# ================= LOAD DATA =================
nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

FILES = {
    "math": [r"..\Datasets\Train\CoT.jsonl", r"..\Datasets\Train\PoT.jsonl"],
    "code": [r"..\Datasets\Train\Python_Code.jsonl", r"..\Datasets\Train\Python_Syntax.jsonl"],
    "general": [r"..\Datasets\Train\Reddit.jsonl"]
}

print("🚜 Loading Data...")
data_frames = []
for category, paths in FILES.items():
    for path in paths:
        try:
            df_cat = pd.read_json(path, lines=True)
            df_cat['category'] = category
            df_cat = df_cat.sample(n=min(len(df_cat), SAMPLES_PER_CLASS // len(paths)), random_state=42)
            data_frames.append(df_cat)
        except ValueError:
            continue

df = pd.concat(data_frames, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print('Constructed dataframe:')
print(df.info())

# ================= CLEAN & VECTORIZE =================
print("🧹 Cleaning & Vectorizing...")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^a-z0-9+=*<> -]', ' ', text.lower())
    words = text.split()
    return ' '.join([ps.stem(w) for w in words if w not in stop_words])
df['clean_text'] = df['input'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 3))
X_numpy = vectorizer.fit_transform(df['clean_text']).toarray()

encoder = LabelEncoder()
Y_numpy = encoder.fit_transform(df['category'])

with open(r"..\VectorStorage\vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)
with open(r"..\VectorStorage\encoder.pkl", "wb") as f: pickle.dump(encoder, f)

# ================= PREPARE TENSORS =================
X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
Y_tensor = torch.tensor(Y_numpy, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ================= DEFINE MODEL =================
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU to training.")
else:
    print("Using CPU for training.")
model = Net(MAX_FEATURES, 3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================= TRAIN LOOP =================
print(f"🚀 Training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f}")

# ================= 6. SAVE MODEL =================
torch.save(model.state_dict(), r"..\Models\Classifier\Router.pth")
print("\n🎉 Training Complete. Model & Vectorizer saved.")