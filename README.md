# Proyecto de Traducción Automática: Inglés → Alemán 🇬🇧->🇩🇪

Este proyecto implementa un sistema de **traducción automática** basado en un modelo **Encoder-Decoder con RNNs (LSTM)**, entrenado con el dataset **Multi30k** para traducir frases del inglés al alemán.

---

##  Teoría Aplicada

Este sistema se basa en el paradigma **seq2seq (sequence-to-sequence)**, comúnmente utilizado en tareas de traducción automática. Su funcionamiento se divide en dos componentes principales:

### 1. Encoder (Codificador)
- Toma una secuencia de entrada en inglés (tokens) y la **convierte en un vector de contexto** que resume el significado de toda la oración.
- Implementado con una red **LSTM** que procesa palabra por palabra y acumula el estado oculto.

### 2. Decoder (Decodificador)
- Recibe el vector de contexto del encoder y **genera secuencialmente la traducción** en alemán.
- También basado en una **LSTM**, genera una palabra a la vez condicionada a las palabras generadas anteriormente.

El entrenamiento se realiza **comparando las salidas generadas con las traducciones reales** mediante **CrossEntropyLoss**, y se optimiza con `Adam`.

---

##  Arquitectura del Proyecto

```
📁 translation_project/
│
├── traslate_en_de.py            # Script principal: entrena, evalúa y permite inferencia interactiva
├── Multi30K_de_en_dataloader.py # Funciones para cargar y preparar el dataset Multi30k
├── requirements.txt             # Todas las dependencias necesarias
├── install_spacy_model.py      # Script para descargar el modelo de tokenización spaCy
└── README.md                    # (Este archivo) Instrucciones y teoría
```

---

##  Recomendación de Instalación en Entorno Virtual (Windows/Linux)

### 1. Clonar o copiar este proyecto

```bash
git clone <URL-del-proyecto>
cd translation_project
```

### 2. Crear un entorno virtual

#### En Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### En Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar el modelo de spaCy para tokenización

```bash
python install_spacy_model.py
```

---

##  Cómo ejecutar el sistema

Entrena el modelo y permite traducir frases desde consola:

```bash
python traslate_en_de.py
```

Cuando finalice el entrenamiento, podrás escribir frases en inglés como:

```
Enter an English sentence to translate: The boy is playing in the park.
→ Output: Der Junge spielt im Park.
```

---

##  Métricas de Evaluación

Durante el entrenamiento se muestra:
- **Pérdida (loss)** por época
- Traducciones reales vs. predichas
- Posibilidad de evaluar con BLEU score

---

##  Notas Finales

- El dataset **Multi30k** contiene ~29K pares de frases inglés-alemán relacionadas con descripciones de imágenes.
- Este es un sistema educativo basado en RNNs. Para tareas industriales se suelen usar Transformers (como BERT o MarianMT).

---

© Ángel Calvar Pastoriza — IBM AI Engineering Labs 2025
