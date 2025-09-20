
# Traducción Automática Inglés → Alemán usando LSTM Seq2Seq (PyTorch)
import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
import numpy as np
import spacy
import gzip
import shutil

def unzip_gz_files(directory):
    for fname in os.listdir(directory):
        if fname.endswith('.gz'):
            path_in = os.path.join(directory, fname)
            path_out = os.path.join(directory, fname[:-3])  # remove .gz
            if not os.path.exists(path_out):
                print(f"Descomprimiendo {fname}...")
                with gzip.open(path_in, 'rt', encoding='utf-8') as f_in, open(path_out, 'w', encoding='utf-8') as f_out:
                    shutil.copyfileobj(f_in, f_out)


# Configuración inicial
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar modelos de tokenización spaCy
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

# Rutas locales de datos
DATA_DIR = './multi30k-dataset/data/task1/raw'
unzip_gz_files(DATA_DIR)  # << descomprime si es necesario

TRAIN_SRC = os.path.join(DATA_DIR, 'train.en')
TRAIN_TRG = os.path.join(DATA_DIR, 'train.de')
VAL_SRC   = os.path.join(DATA_DIR, 'val.en')
VAL_TRG   = os.path.join(DATA_DIR, 'val.de')

# Leer datos
def load_data(src_file, trg_file):
    with open(src_file, encoding='utf-8') as f1, open(trg_file, encoding='utf-8') as f2:
        src_lines = f1.readlines()
        trg_lines = f2.readlines()
    return list(zip(src_lines, trg_lines))

train_data = load_data(TRAIN_SRC, TRAIN_TRG)
val_data = load_data(VAL_SRC, VAL_TRG)

# Construir vocabularios
from collections import Counter

def build_vocab(tokenizer, data, min_freq=2):
    counter = Counter()
    for src, trg in data:
        tokens = tokenizer(src.strip())
        counter.update(tokens)
    vocab = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab

SRC_VOCAB = build_vocab(tokenize_en, train_data, min_freq=1)
TRG_VOCAB = build_vocab(tokenize_de, train_data, min_freq=1)

SRC_itos = {i:s for s,i in SRC_VOCAB.items()}
TRG_itos = {i:s for s,i in TRG_VOCAB.items()}

# Procesamiento de texto
def numericalize(text, vocab, tokenizer):
    tokens = tokenizer(text.strip())
    return [vocab.get('<bos>')] + [vocab.get(tok, vocab['<unk>']) for tok in tokens] + [vocab.get('<eos>')]

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, trg_vocab, src_tok, trg_tok):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tok = src_tok
        self.trg_tok = trg_tok

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        src_tensor = torch.tensor(numericalize(src, self.src_vocab, self.src_tok), dtype=torch.long)
        trg_tensor = torch.tensor(numericalize(trg, self.trg_vocab, self.trg_tok), dtype=torch.long)
        return src_tensor, trg_tensor

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=SRC_VOCAB['<pad>'])
    trg_batch = pad_sequence(trg_batch, padding_value=TRG_VOCAB['<pad>'])
    return src_batch, trg_batch

# DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(TranslationDataset(train_data, SRC_VOCAB, TRG_VOCAB, tokenize_en, tokenize_de),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(TranslationDataset(val_data, SRC_VOCAB, TRG_VOCAB, tokenize_en, tokenize_de),
                        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Modelo Seq2Seq
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# Entrenamiento
INPUT_DIM = len(SRC_VOCAB)
OUTPUT_DIM = len(TRG_VOCAB)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 1
DROPOUT = 0.3

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
PAD_IDX = TRG_VOCAB['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

# Entrenamiento completo
N_EPOCHS = 5
CLIP = 1
train_losses, val_losses = [], []

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    val_loss = evaluate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.3f} | Val Loss = {val_loss:.3f}")

# Visualizar curvas de pérdida
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

# Traducción interactiva
def translate_sentence(sentence, model, src_vocab, trg_vocab, src_tok, max_len=50):
    model.eval()
    tokens = numericalize(sentence, src_vocab, src_tok)
    src_tensor = torch.tensor(tokens).unsqueeze(1).to(device)
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
    trg_indexes = [trg_vocab['<bos>']]
    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab['<eos>']:
            break
    trg_tokens = [TRG_itos[i] for i in trg_indexes[1:-1]]
    return ' '.join(trg_tokens)

print("\nModelo entrenado. Escribe una frase en inglés para traducirla al alemán.")
while True:
    inp = input("Inglés (o 'salir'): ")
    if inp.lower() == 'salir':
        break
    print("Alemán:", translate_sentence(inp, model, SRC_VOCAB, TRG_VOCAB, tokenize_en))
