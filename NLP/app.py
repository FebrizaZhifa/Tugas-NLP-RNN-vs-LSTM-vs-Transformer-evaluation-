import re
from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


DATASET_PATH = Path(__file__).parent / "IMDB_MOVIE_REVIEW.csv"
MAX_LEN = 100
MAX_VOCAB_SIZE = 30_000
EMBED_DIM = 128


@st.cache_data(show_spinner=False)
def load_local_dataset(path: str, modified_time: float) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["review", "label"])


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_data: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_data), usecols=["review", "label"])


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()


def encode_reviews(
    reviews: pd.Series, vocab: dict[str, int]
) -> torch.Tensor:
    encoded = []
    for review in reviews:
        token_ids = [vocab.get(token, 1) for token in clean_text(review).split()]
        token_ids = token_ids[:MAX_LEN]
        encoded.append(token_ids + [0] * (MAX_LEN - len(token_ids)))
    return torch.tensor(encoded, dtype=torch.long)


class BaseModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RNNModel(BaseModel):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int = 64):
        super().__init__(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(self.embedding(inputs))
        return self.sigmoid(self.fc(hidden.squeeze(0)))


class LSTMModel(BaseModel):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int = 64):
        super().__init__(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(self.embedding(inputs))
        return self.sigmoid(self.fc(hidden.squeeze(0)))


class TransformerModel(BaseModel):
    def __init__(
        self, vocab_size: int, embed_dim: int, num_heads: int = 2, num_layers: int = 2
    ):
        super().__init__(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.transformer(self.embedding(inputs))
        return self.sigmoid(self.fc(encoded[:, 0, :]))


MODEL_CLASSES = {
    "RNN": RNNModel,
    "LSTM": LSTMModel,
    "Transformer": TransformerModel,
}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    progress,
    completed_steps: int,
    total_steps: int,
    model_name: str,
) -> tuple[list[float], int]:
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracies = []

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device)).squeeze(1).cpu()
                predictions = outputs > 0.5
                correct += (predictions == labels.bool()).sum().item()
                total += labels.numel()

        accuracies.append(correct / total if total else 0.0)
        completed_steps += 1
        progress.progress(
            completed_steps / total_steps,
            text=f"{model_name} — epoch {epoch + 1}/{epochs}",
        )

    return accuracies, completed_steps


st.set_page_config(page_title="Evaluasi Sentimen IMDB", page_icon="🎬", layout="wide")
st.title("Evaluasi Sentimen Ulasan Film IMDB")
st.write(
    "Bandingkan akurasi model RNN, LSTM, dan Transformer untuk klasifikasi "
    "sentimen ulasan film."
)

uploaded_file = st.sidebar.file_uploader(
    "Dataset CSV (opsional)", type="csv", help="Kolom yang dibutuhkan: review dan label."
)

try:
    if uploaded_file is not None:
        dataset = load_uploaded_dataset(uploaded_file.getvalue())
    elif DATASET_PATH.is_file():
        dataset = load_local_dataset(
            str(DATASET_PATH), DATASET_PATH.stat().st_mtime
        )
    else:
        st.error("Dataset tidak ditemukan. Unggah CSV dengan kolom review dan label.")
        st.stop()
except (
    ValueError,
    KeyError,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
) as error:
    st.error(f"Dataset tidak bisa dibaca. Pastikan kolom review dan label tersedia: {error}")
    st.stop()

dataset["review"] = dataset["review"].fillna("").astype(str)
dataset["label"] = pd.to_numeric(dataset["label"], errors="coerce")
dataset = dataset.dropna(subset=["label"])
dataset = dataset[dataset["label"].isin([0, 1])].copy()
dataset["label"] = dataset["label"].astype("int64")

if dataset.empty or dataset["label"].nunique() != 2:
    st.error("Dataset harus berisi ulasan untuk kedua label: 0 (negatif) dan 1 (positif).")
    st.stop()

class_counts = dataset["label"].value_counts()
if class_counts.min() < 2:
    st.error("Setiap label harus memiliki setidaknya dua ulasan.")
    st.stop()

st.caption(
    f"Dataset: {len(dataset):,} ulasan valid · "
    f"Negatif: {class_counts.get(0, 0):,} · Positif: {class_counts.get(1, 0):,}"
)

with st.sidebar:
    st.header("Pengaturan eksperimen")
    sample_size = st.number_input(
        "Jumlah ulasan",
        min_value=2,
        max_value=len(dataset),
        value=min(3_000, len(dataset)),
        step=min(500, max(1, len(dataset) - 1)),
        help="Memakai sebagian data mempercepat training di Streamlit Cloud.",
    )
    epochs = st.slider("Epoch", min_value=1, max_value=5, value=2)
    selected_models = st.multiselect(
        "Model yang dibandingkan",
        options=list(MODEL_CLASSES),
        default=list(MODEL_CLASSES),
    )

if st.button("Mulai evaluasi", type="primary", disabled=not selected_models):
    torch.manual_seed(42)
    try:
        sampled = dataset
        if sample_size < len(dataset):
            sampled, _ = train_test_split(
                dataset,
                train_size=int(sample_size),
                stratify=dataset["label"],
                random_state=42,
            )
        train_df, val_df = train_test_split(
            sampled,
            test_size=0.2,
            stratify=sampled["label"],
            random_state=42,
        )
    except ValueError as error:
        st.error(f"Data belum cukup untuk dibagi secara stratifikasi: {error}")
        st.stop()

    token_counts = Counter(
        token
        for review in train_df["review"]
        for token in clean_text(review).split()
    )
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update(
        {
            token: index + 2
            for index, (token, _) in enumerate(token_counts.most_common(MAX_VOCAB_SIZE))
        }
    )

    train_inputs = encode_reviews(train_df["review"], vocab)
    val_inputs = encode_reviews(val_df["review"], vocab)
    train_labels = torch.tensor(train_df["label"].to_numpy(), dtype=torch.float32)
    val_labels = torch.tensor(val_df["label"].to_numpy(), dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_labels), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(val_inputs, val_labels), batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Training di perangkat: {device.type.upper()}")
    progress = st.progress(0.0, text="Menyiapkan training...")
    histories = {}
    completed_steps = 0
    total_steps = len(selected_models) * epochs

    for model_name in selected_models:
        model = MODEL_CLASSES[model_name](len(vocab), EMBED_DIM)
        histories[model_name], completed_steps = train_model(
            model,
            train_loader,
            val_loader,
            epochs,
            device,
            progress,
            completed_steps,
            total_steps,
            model_name,
        )

    st.session_state["histories"] = histories
    st.session_state["sample_size"] = len(sampled)

if "histories" in st.session_state:
    histories = st.session_state["histories"]
    st.subheader("Hasil evaluasi")
    results = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Akurasi validasi terakhir": f"{values[-1]:.2%}",
            }
            for model_name, values in histories.items()
        ]
    )
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.caption(
        f"Menggunakan {st.session_state['sample_size']:,} ulasan "
        "dengan pembagian training/validasi 80:20."
    )

    figure, axis = plt.subplots(figsize=(9, 4))
    for model_name, values in histories.items():
        axis.plot(range(1, len(values) + 1), values, marker="o", label=model_name)
    axis.set(title="Akurasi validasi per epoch", xlabel="Epoch", ylabel="Akurasi")
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
