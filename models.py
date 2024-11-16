# models.py

import numpy as np
import collections
from utils import Indexer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

#####################
# RNN Classifier #
#####################

class RNNClassifier(ConsonantVowelClassifier, nn.Module):  # Ensure correct inheritance
    def __init__(self, vocab_size, embed_dim = 256, hidden_dim = 512, output_dim=2, vocab_index=None, num_layers=3):
        super(RNNClassifier, self).__init__()  # Initialize both parent classes
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.vocab_index = vocab_index  # Store vocab_index as an attribute

    def forward(self, x):
        embedded = self.embedding(x)  # Shape: (batch_size, seq_length, embed_dim)
        _, hidden = self.gru(embedded)  # hidden: (num_layers, batch_size, hidden_dim)
        hidden = hidden[-1]  # Get the last hidden state from the last layer (batch_size, hidden_dim)
    
        output = self.fc(hidden)  # Pass through the fully connected layer (batch_size, output_dim)
        return output
  # Return raw logits without softmax

    def predict(self, context):
        input_tensor = string_to_tensor(context, self.vocab_index).unsqueeze(0)  # Convert string to tensor
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class


vocab_index = Indexer()
for char in "abcdefghijklmnopqrstuvwxyz ":
    vocab_index.add_and_get_index(char)




def string_to_tensor(s, vocab_index, max_length=20):
    """
    Converts a raw string to a PyTorch tensor of indices based on the vocab_index,
    truncating or padding to max_length.
    """
    s = s[:max_length]  # Truncate to max_length if necessary
    indices = [vocab_index.index_of(char) for char in s]
    if len(indices) < max_length:
        # Pad with index 0 (assuming 0 is the padding index)
        indices += [0] * (max_length - len(indices))
    
    tensor = torch.tensor(indices, dtype=torch.long)
    return tensor

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)  # Pad inputs
    labels = torch.stack(labels)  # Stack labels into a tensor
    return inputs, labels


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    Trains an RNNClassifier on the provided training data.

    :param args: Command-line args
    :param train_cons_exs: List of training sequences followed by consonants
    :param train_vowel_exs: List of training sequences followed by vowels
    :param dev_cons_exs: List of dev sequences followed by consonants
    :param dev_vowel_exs: List of dev sequences followed by vowels
    :param vocab_index: Indexer of the character vocabulary (27 characters)
    :return: Trained RNNClassifier instance
    """
    # Hyperparameters
    vocab_size = len(vocab_index)
    embed_dim = getattr(args, 'embed_dim', 128)
    hidden_dim = getattr(args, 'hidden_dim', 256)
    output_dim = 2 
    num_layers = getattr(args, 'num_layers', 1)
    learning_rate = getattr(args, 'learning_rate', 0.001)
    num_epochs = getattr(args, 'num_epochs', 10)
    batch_size = getattr(args, 'batch_size', 32)

    # Initialize the model
    model = RNNClassifier(vocab_size, embed_dim, hidden_dim, output_dim, vocab_index, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare training data
    train_data = [(string_to_tensor(s, vocab_index), 0) for s in train_cons_exs] + \
                 [(string_to_tensor(s, vocab_index), 1) for s in train_vowel_exs]

    # DataLoader with updated collate_fn
    train_tensors = [(s, torch.tensor(label, dtype=torch.long)) for s, label in train_data]
    train_loader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:  # Unpack batch

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)  # Model forward pass (batch_size x num_classes)
    

            # Compute loss
            loss = criterion(outputs, labels)  # Ensure outputs and labels align
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

    return model

#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

    def get_log_prob_single(self, next_char, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    raise Exception("Implement me")
