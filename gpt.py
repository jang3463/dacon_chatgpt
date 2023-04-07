import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import torch.nn.functional as F

class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["label"]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }

def train_epoch(model, data_loader, optimizer, scheduler, device, loss_function):
    model = model.train()
    losses = []
    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_function(logits, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

    return np.mean(losses)

def evaluate(model, data_loader, device, loss_function):
    model = model.eval()
    labels_all = []
    predictions_all = []
    losses = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_function(logits, labels)
            losses.append(loss.item())

            _, preds = torch.max(logits, dim=1)
            labels_all.extend(labels.cpu().numpy())
            predictions_all.extend(preds.cpu().numpy())

    loss_mean = np.mean(losses)
    f1_macro = f1_score(labels_all, predictions_all, average='macro')
    return loss_mean, f1_macro

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=8).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 10
batch_size = 8

train_data = pd.read_csv("train.csv")
train_data, val_data = train_test_split(train_data, test_size=0.1, stratify=train_data["label"], random_state=42)

train_dataset = NewsDataset(train_data, tokenizer, max_length=512)
val_dataset = NewsDataset(val_data, tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

total_steps = len(train_loader) * num_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps,
)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss
        
# Instantiate the FocalLoss
# loss_function = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean").to(device)
loss_function = FocalLoss(alpha=1.0, gamma=3.0, reduction="mean").to(device)

best_f1_macro = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 10)

    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_function)
    print(f"Train loss: {train_loss}")

    val_loss, val_f1_macro = evaluate(model, val_loader, device, loss_function)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation F1 Macro: {val_f1_macro}")

    if val_f1_macro > best_f1_macro:
        print("F1 Macro score improved. Saving the model.")
        best_f1_macro = val_f1_macro
        torch.save(model.state_dict(), "best_roberta_large_model.bin")

print("Training completed.")

def predict(model, data_loader, device):
    model = model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions

# Load the best model
model.load_state_dict(torch.load("best_roberta_large_model.bin"))

class TestNewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

# Load and preprocess the test dataset
test_data = pd.read_csv("test.csv")
test_dataset = TestNewsDataset(test_data, tokenizer, max_length=512)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Make predictions on the test dataset
predictions = predict(model, test_loader, device)

# Create submission DataFrame and save it as a CSV file
submission = pd.DataFrame({"id": test_data["id"], "label": predictions})
submission.to_csv("submission.csv", index=False)

print("Inference completed. The submission.csv file has been generated.")
