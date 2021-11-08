import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from collections import defaultdict

from transformers import XLNetTokenizer
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import AlbertTokenizer
from transformers import XLNetForSequenceClassification
from transformers import RobertaForSequenceClassification
from transformers import AlbertForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np
from torch.optim import Adam
from torch import nn, optim
import torch.nn.functional as F
import time
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import classification_report
import os
import argparse


data_path =''
test_data_path=''
results_rootpath=''
modelName =''
pretrained_model_path=''
batch_size = 24
MAX_SEQ_LEN= 234
EPOCHS = 3

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", help="Name of the model to be used")
args = parser.parse_args()
if args.model:
    modelName = args.model

print('Model Name is:', modelName)
def Get_Model(modelName):
    model=''
    if modelName == 'XLNet':
        model = XLNetForSequenceClassification.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            pretrained_model_path,
            # The number of output labels--2 for binary classification.
            num_labels=2
        )
    elif modelName == 'BERT':
        model = BertForSequenceClassification.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            pretrained_model_path,
            # The number of output labels--2 for binary classification.
            num_labels=2
        )
    elif modelName == 'RoBerta':
        model = RobertaForSequenceClassification.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            pretrained_model_path,
            # The number of output labels--2 for binary classification.
            num_labels=2
        )
    elif modelName == 'Albert':
        model = AlbertForSequenceClassification.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            pretrained_model_path,
            # The number of output labels--2 for binary classification.
            num_labels=2
        )
    return model

def Get_Encodings(modelName, inputText):   
    input_ids = []
    attention_masks = []
    if modelName == 'XLNet':  
        tokenizer = XLNetTokenizer.from_pretrained(pretrained_model_path, do_lower_case=True)
    elif modelName == 'BERT':
        tokenizer = BERTTokenizer.from_pretrained(
            pretrained_model_path, do_lower_case=True)
    elif modelName == 'RoBerta':
        tokenizer = RobertaTokenizer.from_pretrained(retrained_model_path, do_lower_case=True)
    elif modelName == 'Albert':
        tokenizer = AlbertTokenizer.from_pretrained(retrained_model_path, do_lower_case=True)

    for sent in inputText:   
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_SEQ_LEN,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt')  # Return pytorch tensors.
        input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    return input_ids, attention_masks


def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    acc = 0
    counter = 0

    for d in data_loader:
        input_ids = d["input_ids"].reshape(batch_size, MAX_SEQ_LEN).to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, token_type_ids=None,
                        attention_mask=attention_mask, labels=targets)
        loss = outputs[0]
        logits = outputs[1]

        # preds = preds.cpu().detach().numpy()
        _, prediction = torch.max(outputs[1], dim=1)
        targets = targets.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, prediction)

        acc += accuracy
        losses.append(loss.item())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        counter = counter + 1

    return acc / counter, np.mean(losses)


def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    acc = 0
    counter = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].reshape(
                batch_size, MAX_SEQ_LEN).to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, token_type_ids=None,
                            attention_mask=attention_mask, labels=targets)
            loss = outputs[0]
            logits = outputs[1]

            _, prediction = torch.max(outputs[1], dim=1)
            targets = targets.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = metrics.accuracy_score(targets, prediction)

            acc += accuracy
            losses.append(loss.item())
            counter += 1

    return acc / counter, np.mean(losses)


def get_predictions(model, data_loader):
    model = model.eval()

    tweet_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["texts"]
            input_ids = d["input_ids"].reshape(batch_size, MAX_SEQ_LEN).to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, token_type_ids=None,
                            attention_mask=attention_mask, labels=targets)

            loss = outputs[0]
            logits = outputs[1]

            _, preds = torch.max(outputs[1], dim=1)

            probs = F.softmax(outputs[1], dim=1)

            tweet_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return tweet_texts, predictions, prediction_probs, real_values

df = pd.read_csv(data_path, sep=",", encoding="utf-8",names=['tweet_id', 'texts', 'labels', 'disease'],  usecols=range(4))
df_train, df_valid = train_test_split(df, test_size=0.5, random_state=101)
df_train = shuffle(df_train)

df_test = pd.read_csv(test_data_path, sep=",", encoding="utf-8", names=['tweet_id', 'texts', 'labels', 'disease'],  usecols=range(4))
print('Train Df -lenght: ', len(df_train))
print('Valid Df -lenght: ', len(df_valid))
print('Test Df -lenght: ', len(df_test))
train_sentences = df_train.texts.to_list()
df_train['labels'].replace(1, 0, inplace=True)
df_train['labels'].replace(2, 1, inplace=True)

df_valid['labels'].replace(1, 0, inplace=True)
df_valid['labels'].replace(2, 1, inplace=True)

df_test['labels'].replace(1, 0, inplace=True)
df_test['labels'].replace(2, 1, inplace=True)

train_labels = df_train.labels.to_list()

valid_sentences = df_valid.texts.to_list()
valid_labels = df_valid.labels.to_list()

test_sentences = df_test.texts.to_list()
test_labels = df_test.labels.to_list()

input_ids, attention_masks = Get_Encodings(modelName, train_sentences)
train_input_ids = torch.cat(input_ids, dim=0)
train_attention_masks = torch.cat(attention_masks, dim=0)
train_labels = torch.tensor(train_labels)

valid_input_ids, valid_attention_masks = Get_Encodings(
    modelName, valid_sentences)
valid_input_ids = torch.cat(valid_input_ids, dim=0)
valid_attention_masks = torch.cat(valid_attention_masks, dim=0)
valid_labels = torch.tensor(valid_labels)



# Combine the training inputs into a TensorDataset.
train_dataset = TensorDataset(
    train_input_ids, train_attention_masks, train_labels)

valid_dataset = TensorDataset(
    valid_input_ids, valid_attention_masks, valid_labels)




# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size  # Trains with this batch size.
)

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
    valid_dataset,  # The validation samples.
    sampler=SequentialSampler(valid_dataset),  # Pull out batches sequentially.
    batch_size=batch_size  # Evaluate with this batch size.
)

model = Get_Model(modelName)

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model = model.to(device)




param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay':0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print('Epoch No: ', str(epoch+1))


    train_acc, train_loss = train_epoch(
        model,
        train_dataloader,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )

    print('Train loss:', str(train_loss),' Train accuracy: ', str(train_acc))

    val_acc, val_loss = eval_model(
        model,
        validation_dataloader,
        device,
        len(df_valid)
    )

    print('Val loss:', str(val_loss), ' Val accuracy: ', str(val_acc))

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(
        ), '/fine_tuned/xlnet_model.bin')
        best_accuracy = val_acc


pd.DataFrame(history).to_csv("history.csv")
test_input_ids, test_attention_masks = Get_Encodings(modelName, test_sentences)
test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_labels = torch.tensor(test_labels)

test_dataset = TensorDataset(
    test_input_ids, test_attention_masks, test_labels)

test_dataloader = DataLoader(
    test_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size  # Trains with this batch size.
)

model.load_state_dict(torch.load('/fine_tuned/xlnet_model.bin'))

model = model.to(device)

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    test_dataloader
)
report = classification_report(y_test, y_pred)


# Save the report into file
output_eval_file = os.path.join(results_rootpath, "eval_results-Fold1.txt")
with open(output_eval_file, "w") as writer:
    print("***** Eval results *****")

    print(report)
    writer.write("\n\n")
    writer.write(report)
