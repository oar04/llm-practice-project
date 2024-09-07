# Bigram Language Model using PyTorch

This project implements a simple Bigram Language Model using PyTorch. The model is trained on a text dataset, specifically "The Wizard of Oz," to predict the next character in a sequence based on the current one.

## Project Overview

The goal of this project is to develop a basic bigram model, where the model learns the probabilities of the next character given the current character. This model serves as an introduction to language modeling and can be extended to more complex architectures like trigrams, LSTMs, or Transformers.

## Features

- Loads and processes text data
- Maps characters to integers and vice versa
- Splits the dataset into training and validation sets
- Trains a simple bigram model for character-level prediction
- Can generate text based on a seed input after training

## Prerequisites

Make sure you have the following installed before running the project:

- Python 3.7+
- PyTorch
- NumPy

You can install the required packages using the following command:

```bash
pip install torch numpy

## Dataset

The project uses a text file from "The Wizard of Oz" as the dataset. The text is read from `wizard_of_oz.txt`, and characters are converted into integers for processing by the model.

## Code Overview

### Key Sections

1. **Device Setup**: Checks if a GPU is available and sets the device accordingly:

    ```python
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ```

2. **Data Loading and Preprocessing**: Loads the dataset and prepares mappings for characters to integers and vice versa.

    ```python
    with open('data/wizard_of_oz.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    string_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_string = {i: ch for i, ch in enumerate(chars)}
    ```

3. **Train-Validation Split**: Splits the data into training (80%) and validation (20%) sets.

    ```python
    n = int(0.8 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    ```

4. **Model Input Preparation**: Prepares the data in the form of input-output pairs for the bigram model:

    ```python
    x = train_data[:block_size]
    y = train_data[1:block_size + 1]
    ```

### Future Work

- Implement the model architecture using an embedding layer followed by a simple linear layer for predicting the next character.
- Train the model using cross-entropy loss and optimize with Adam or SGD.
- Generate text after training using a seed character and the trained model.
- Evaluate the model on the validation set.

