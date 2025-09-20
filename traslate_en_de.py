# --------------------------------------------------
# SCRIPT COMPLETO - Seq2Seq (EN->DE) con aceleración MPS y guardado
# --------------------------------------------------

import os
import random
import math
import time
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------
# Configuración del dispositivo
# -------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Usando:", device)

# -------------------------
# Tokenización
# -------------------------
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

# -------------------------
# Carga del dataset
# -------------------------
DATA_DIR = './multi30k-dataset/data/task1/raw'
TRAIN_SRC = os.path.join(DATA_DIR, 'train.en')
TRAIN_TRG = os.path.join(DATA_DIR, 'train.de')
VAL_SRC   = os.path.join(DATA_DIR, 'val.en')
VAL_TRG   = os.path.join(DATA_DIR, 'val.de')

def load_data(src_file, trg_file):
    with open(src_file, encoding='utf-8') as f1, open(trg_file, encoding='utf-8') as f2:
        src_lines = f1.readlines()
        trg_lines = f2.readlines()
    return list(zip(src_lines, trg_lines))

train_data = load_data(TRAIN_SRC, TRAIN_TRG)
val_data = load_data(VAL_SRC, VAL_TRG)

# -------------------------
# Preprocesamiento
# -------------------------
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'de'

specials = ['<unk>', '<pad>', '<bos>', '<eos>']

# Tokenizer wrappers
def yield_tokens(data_iter, language):
    for src, trg in data_iter:
        text = src if language == SRC_LANGUAGE else trg
        tokenizer = tokenize_en if language == SRC_LANGUAGE else tokenize_de
        yield tokenizer(text)

# Construcción vocabulario
vocab_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    tokenizer = tokenize_en if ln == SRC_LANGUAGE else tokenize_de
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_data, ln),
                                                    specials=specials,
                                                    min_freq=2)
    vocab_transform[ln].set_default_index(vocab_transform[ln]['<unk>'])

# Pipelines
text_transform = {
    SRC_LANGUAGE: lambda x: [vocab_transform[SRC_LANGUAGE]['<bos>']] + 
                             [vocab_transform[SRC_LANGUAGE][tok] for tok in tokenize_en(x)] + 
                             [vocab_transform[SRC_LANGUAGE]['<eos>']],

    TRG_LANGUAGE: lambda x: [vocab_transform[TRG_LANGUAGE]['<bos>']] + 
                             [vocab_transform[TRG_LANGUAGE][tok] for tok in tokenize_de(x)] + 
                             [vocab_transform[TRG_LANGUAGE]['<eos>']]
}

# Collate para DataLoader
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(torch.tensor(text_transform[SRC_LANGUAGE](src_sample.rstrip())))
        trg_batch.append(torch.tensor(text_transform[TRG_LANGUAGE](trg_sample.rstrip())))
    src_batch = pad_sequence(src_batch, padding_value=vocab_transform[SRC_LANGUAGE]['<pad>'])
    trg_batch = pad_sequence(trg_batch, padding_value=vocab_transform[TRG_LANGUAGE]['<pad>'])
    return src_batch, trg_batch

# Datasets y Dataloaders
BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -------------------------
# Definición del Modelo
# -------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input = trg[0,:]  # <bos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs

# -------------------------
# Inicialización del modelo
# -------------------------
INPUT_DIM = len(vocab_transform[SRC_LANGUAGE])
OUTPUT_DIM = len(vocab_transform[TRG_LANGUAGE])
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=vocab_transform[TRG_LANGUAGE]['<pad>'])

# -------------------------
# Funciones de entrenamiento y evaluación
# -------------------------
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(iterator, desc="Training"):
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
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Evaluating"):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # sin teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# -------------------------
# Entrenamiento
# -------------------------
N_EPOCHS = 10
CLIP = 1
MODEL_PATH = 'RNN-TR-model.pt'

if os.path.exists(MODEL_PATH):
    print("Cargando modelo entrenado desde disco...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("Entrenando modelo...")
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_loader, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"Epoch {epoch+1}: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}")

# -------------------------
# Inferencia interactiva
# -------------------------
def translate_sentence(model, sentence):
    model.eval()
    tokens = tokenize_en(sentence.lower().strip())
    indices = [vocab_transform[SRC_LANGUAGE]['<bos>']] + [vocab_transform[SRC_LANGUAGE][tok] for tok in tokens] + [vocab_transform[SRC_LANGUAGE]['<eos>']]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [vocab_transform[TRG_LANGUAGE]['<bos>']]

    for _ in range(50):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == vocab_transform[TRG_LANGUAGE]['<eos>']:
            break

    trg_tokens = [vocab_transform[TRG_LANGUAGE].lookup_token(i) for i in trg_indexes]
    return ' '.join(trg_tokens[1:-1])  # sin <bos> ni <eos>

print("\nModelo listo. Escribe una frase en inglés para traducirla al alemán. (escribe 'salir' para terminar)")
while True:
    user_input = input("Inglés: ")
    if user_input.lower() in ["salir", "exit"]:
        break
    german = translate_sentence(model, user_input)
    print("Alemán:", german)

