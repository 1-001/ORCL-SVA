import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import RobertaTokenizer, T5EncoderModel, AdamW, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
from pacing_functions import PACING_FUNCTIONS
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Example:
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target


class InputFeatures:
    def __init__(self, example_id, source_ids, target_id, source_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_id = target_id
        self.source_mask = source_mask

def read_examples(filename, is_train=False):
    data = pd.read_excel(filename).astype(str)
    if is_train:
        data = data.sort_values(by='score')  
    examples = []
    for idx, row in data.iterrows():
        source = f"{row['abstract_func_before']} <desc> {row['description']}"
        target = int(row['severity'])
        examples.append(Example(idx=idx, source=source, target=target))
    return examples


def convert_examples_to_features(examples, tokenizer, max_length):
    features = []
    for example in examples:
        tokens = tokenizer(example.source, max_length=max_length, padding='max_length', truncation=True)
        features.append(InputFeatures(
            example_id=example.idx,
            source_ids=tokens['input_ids'],
            target_id=example.target,
            source_mask=tokens['attention_mask']
        ))
    return features


class CORALOrdinalLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(CORALOrdinalLoss, self).__init__()
        self.num_classes = num_classes - 1

    def forward(self, logits, labels):
        labels_bin = torch.zeros(labels.size(0), self.num_classes).to(logits.device)
        for i in range(self.num_classes):
            labels_bin[:, i] = (labels > i).float()
        loss_func = torch.nn.BCEWithLogitsLoss()
        loss = loss_func(logits, labels_bin)
        return loss


class CORALPromptModelWithLinear(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super(CORALPromptModelWithLinear, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.num_classes = num_classes - 1
        self.linear = torch.nn.Linear(self.encoder.config.d_model, self.num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.linear(cls_output)
        return logits


def evaluate(model, dataloader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            source_ids, source_mask, target_ids = [x.to(device) for x in batch]
            logits = model(input_ids=source_ids, attention_mask=source_mask)
            preds = (logits > 0).sum(dim=-1).cpu().tolist()
            predictions.extend(preds)
            labels.extend(target_ids.cpu().tolist())

    acc = accuracy_score(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)
    mse = np.mean([(p - l) ** 2 for p, l in zip(predictions, labels)])
    mze = np.mean([p != l for p, l in zip(predictions, labels)])
    macro_f1 = f1_score(labels, predictions, average='macro')
    mcc = matthews_corrcoef(labels, predictions)
    return acc, spearman_corr, mse, mze, macro_f1, mcc


def main():
    model_name = "E:\\models\\codet5-base"
    train_file = r"E:\wjy\ordinal regression\code\complete_test_data_with_absolute_errors.xlsx"
    val_file = r"C:\Users\Admin\Desktop\data_c++\valid.xlsx"
    test_file = r"C:\Users\Admin\Desktop\data_c++\test.xlsx"
    output_dir = "./output"
    max_length = 512
    num_classes = 4
    num_epochs = 100
    batch_size = 16
    learning_rate = 5e-5
    patience = 10
    c0 = 0.33
    all_step = 0
    set_seed(42)


    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = CORALPromptModelWithLinear(model_name, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    train_examples = read_examples(train_file, is_train=True)
    val_examples = read_examples(val_file, is_train=False)
    test_examples = read_examples(test_file, is_train=False)


    val_features = convert_examples_to_features(val_examples, tokenizer, max_length)
    test_features = convert_examples_to_features(test_examples, tokenizer, max_length)

    val_dataloader = DataLoader(
        TensorDataset(
            torch.tensor([f.source_ids for f in val_features], dtype=torch.long),
            torch.tensor([f.source_mask for f in val_features], dtype=torch.long),
            torch.tensor([f.target_id for f in val_features], dtype=torch.long),
        ),
        sampler=SequentialSampler(val_features),
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        TensorDataset(
            torch.tensor([f.source_ids for f in test_features], dtype=torch.long),
            torch.tensor([f.source_mask for f in test_features], dtype=torch.long),
            torch.tensor([f.target_id for f in test_features], dtype=torch.long),
        ),
        sampler=SequentialSampler(test_features),
        batch_size=batch_size
    )

    coral_loss_func = CORALOrdinalLoss(num_classes)
    subset_percentage = c0

    while subset_percentage <= 1.0:
        
        subset_length = int(len(train_examples) * subset_percentage)
        train_subset = train_examples[:subset_length]
        train_features = convert_examples_to_features(train_subset, tokenizer, max_length)
    
        train_dataloader = DataLoader(
            TensorDataset(
                torch.tensor([f.source_ids for f in train_features], dtype=torch.long),
                torch.tensor([f.source_mask for f in train_features], dtype=torch.long),
                torch.tensor([f.target_id for f in train_features], dtype=torch.long),
            ),
            sampler=RandomSampler(train_features),
            batch_size=batch_size
        )
    
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=len(train_dataloader) * num_epochs, num_training_steps=len(train_dataloader) * num_epochs
        )
    
        best_metric = float("inf")
        early_stop_count = 0
    
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
                source_ids, source_mask, target_ids = [x.to(device) for x in batch]
                optimizer.zero_grad()
                logits = model(input_ids=source_ids, attention_mask=source_mask)
                loss = coral_loss_func(logits, target_ids)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                all_step += 1
    
                if (step + 1) % 100 == 0:
                    avg_loss = total_loss / (step + 1)
                    print(f"Subset {subset_percentage:.2f}, Epoch {epoch + 1}, Step {step + 1}, Loss: {avg_loss:.4f}")
    
            acc, spearman_corr, mse, mze, macro_f1, mcc = evaluate(model, val_dataloader, device)
            print(f"Validation ACC: {acc:.4f}, F1: {macro_f1:.4f}, MSE: {mse:.4f}, MZE: {mze:.4f}, MCC: {mcc:.4f}")
    
            if mse < best_metric:
                best_metric = mse
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                print(f"New best model saved with MSE: {best_metric:.4f}")
                early_stop_count = 0
            else:
                early_stop_count += 1
                print(f"No improvement. Early stop count: {early_stop_count}/{patience}")
    
            if early_stop_count >= patience:
                print("Early stopping triggered for this subset.")
                break
    
        
        total_steps = num_epochs * len(train_dataloader)/2
        subset_percentage = min(1, PACING_FUNCTIONS["root_10"](all_step, total_steps, c0))
        print(f"Updated subset to {subset_percentage:.2f} of data.")
        
        if subset_percentage >= 1.0 and early_stop_count == 10:
            print("Training on all data subsets complete.")
            break  
    
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    acc, spearman_corr, mse, mze, macro_f1, mcc = evaluate(model, test_dataloader, device)
    print(f"Test Results: ACC: {acc:.4f}, F1: {macro_f1:.4f}, MSE: {mse:.4f}, MZE: {mze:.4f}, MCC: {mcc:.4f}")

if __name__ == "__main__":
    main()
