import os
import mlflow
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW

params = {
    'model_name': 'distilbert-base-cased',
    'learning_rate': 5e-5,
    'batch_size': 16,
    'num_epochs': 1,
    'dataset_name': 'ag_news',
    'task_name': 'sequence_classification',
    'log_steps': 100,
    'max_seq_length': 128,
    'output_dir': 'models/distilbert-base-uncased-ag_news',
}

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"{params['task_name']}")

# Load and preprocess dataset
dataset = load_dataset(params['dataset_name'])#, params['task_name'])
tokenizer = DistilBertTokenizer.from_pretrained(params['model_name'])

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=params['max_seq_length'])


train_dataset = dataset["train"].shuffle().select(range(20_000)).map(tokenize, batched=True)
test_dataset = dataset["test"].shuffle().select(range(2_000)).map(tokenize, batched=True)

# Set format for PyTorch and create data loaders
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

# get the labels
labels = dataset["train"].features['label'].names

# model init
model = DistilBertForSequenceClassification.from_pretrained(params['model_name'], 
                                                            num_labels=len(labels))
model.config.id2label = {i: label for i, label in enumerate(labels)}
params['id2label'] = model.config.id2label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer
optimizer = AdamW(model.parameters(), lr=params['learning_rate'])

# evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)

            # Forward pass, calculate logit predictions
            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits
            _, predicted_labels = torch.max(logits, dim=1)

            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate Evaluation Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')

    return accuracy, precision, recall, f1

# Start MLflow Run
with mlflow.start_run(run_name=f"{params['model_name']}-{params['dataset_name']}") as run:

    # Log all parameters at once
    mlflow.log_params(params)

    with tqdm(total=params['num_epochs'] * len(train_loader), desc=f"Epoch [1/{params['num_epochs']}] - (Loss: N/A) - Steps") as pbar:
        for epoch in range(params['num_epochs']):
            running_loss = 0.0
            for i, batch in enumerate(train_loader, 0):
                inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(inputs, attention_mask=masks, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i and i % params['log_steps'] == 0:
                    avg_loss = running_loss / params['log_steps']

                    pbar.set_description(f"Epoch [{epoch + 1}/{params['num_epochs']}] - (Loss: {avg_loss:.3f}) - Steps")
                    mlflow.log_metric("loss", avg_loss, step=epoch * len(train_loader) + i)

                    running_loss = 0.0
                pbar.update(1)

            # Evaluate Model
            accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
            print(f"Epoch {epoch + 1} Metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, step=epoch)


    # Log model to MLflow through built-in PyTorch method
    # mlflow.pytorch.log_model(model, "model")

    # Log model to MLflow through custom method
    os.makedirs(params['output_dir'], exist_ok=True)
    model.save_pretrained(params['output_dir'])
    tokenizer.save_pretrained(params['output_dir'])

    mlflow.log_artifacts(params['output_dir'], artifact_path="model")

    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "agnews-transformer")

print('Finished Training')