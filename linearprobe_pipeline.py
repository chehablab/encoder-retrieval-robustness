from encoders import get_encoder, get_features
from datasets import get_dataset
from torch.utils.data.dataloader import DataLoader
import torch 
import os
from time import time
from tqdm.notebook import tqdm
import numpy as np

def save_checkpoint(path, classifier, optimizer, epoch, history, hyperparams, early_stopped, weights_only=False):
    checkpoint = {
        'classifier_state': classifier.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'history': history,
        'hyperparams': hyperparams,
        'early_stopped': early_stopped
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, classifier, optimizer):
    checkpoint = torch.load(path, weights_only=False)
    classifier.load_state_dict(checkpoint['classifier_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint['epoch']
    history = checkpoint['history']
    return classifier, optimizer, epoch, history

def probe(encoder_name, dataset_name, batch_size= 64, n_epochs= 20,
          encoder_target_dim=768, num_workers=4, learning_rate=1e-3,
          early_stopping_based_on_validation=False,
          early_stopping_based_on_test=True,
          random_state=42, chkpt_path="./chkpt",
          test_every_x_steps=1, validate=False,
          verbose=True):
    
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    
    # Save hyperparameters
    hyperparams = {
        "batch_size": batch_size,
        "encoder_target_dim": encoder_target_dim,
        "learning_rate": learning_rate,
    }
    
    # Create checkpoint directory
    if not os.path.exists(chkpt_path):
        os.mkdir(chkpt_path)

    # Get encoder
    if verbose: print("Loading model ...")
    encoder, processor = get_encoder(encoder_name)

    # Get device
    device = next(encoder.parameters()).device

    # Get datasets
    if verbose: print("Loading dataset ...")
    train_dataset = get_dataset(dataset_name, "train", processor)
    test_dataset = get_dataset(dataset_name, "test", processor)
    val_dataset = get_dataset(dataset_name, "val", processor)

    # Get dataloaders
    if verbose: print("Loading dataloaders ...")
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle= True, num_workers= num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle= False, num_workers= num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle= False, num_workers= num_workers)

    # Define classifier
    if verbose: print("Defining classifier ...")
    classifier = torch.nn.Linear(encoder_target_dim, train_dataset.num_labels())
    classifier.to(device)

    # Define optimizer
    if verbose: print("Defining optimizer ...")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    # Load checkpoint
    if verbose: print("Loading checkpoint ...")
    escaped_encoder_name = encoder_name.replace("/", "_")
    escaped_dataset_name = dataset_name.replace("/", "_")
    chkpt_filename = f"{escaped_encoder_name}_{escaped_dataset_name}.pt"
    chkpt_filepath = os.path.join(chkpt_path, chkpt_filename)
    if os.path.exists(chkpt_filepath):
        classifier, optimizer, start_epoch, history = load_checkpoint(chkpt_filepath, classifier, optimizer) 
    else:
        start_epoch = 0
        history = []

    # Define criterion
    if verbose: print("Defining criterion ...")
    if train_dataset.is_multilabel():
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if verbose: print("Starting training ...")
    for epoch in range(start_epoch, n_epochs):
        train_losses = []

        classifier.train()
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                features = get_features(encoder, inputs, encoder_target_dim, device=device)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({"Train Loss": loss.item()})

        train_loss = sum(train_losses) / len(train_losses)
        tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}")

        if validate:
            # Validation loop
            classifier.eval()
            val_losses = []
            val_preds = []
            val_labels = []
            pbar = tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1}/{n_epochs}')
            for batch in pbar:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)  
                
                with torch.no_grad():
                    features = get_features(encoder, inputs, encoder_target_dim, device=device)
                    outputs = classifier(features)

                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                pbar.set_postfix({"Val Loss": loss.item()})

                if train_dataset.is_multilabel():
                    predicted = (torch.sigmoid(outputs) > 0.5).int()
                    val_preds.extend(predicted.flatten().cpu().numpy())
                    val_labels.extend(labels.flatten().cpu().numpy())
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_loss = sum(val_losses) / len(val_losses)
            val_acc = 100.0 * (np.array(val_preds) == np.array(val_labels)).sum() / len(val_labels)
            tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

            # Early stopping
            if early_stopping_based_on_validation:
                recent_accs = [val_record["val_accuracy"] for val_record in history[-5:]]
                recent_accs.append(val_acc)
                # If no improvement in last 6 epochs, stop training
                # 6 epochs to have at least one test iteration
                if (len(recent_accs) >= 6) and (max(recent_accs) - min(recent_accs) < 0.05):
                    print("Early stopping triggered. No improvement in validation accuracy.")
                    history.append({
                        "epoch": epoch + 1, 
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "test_loss": None,
                        "test_accuracy": None
                    })
                    save_checkpoint(chkpt_filepath, classifier, optimizer, epoch + 1, history, hyperparams, early_stopped=True)
                    break
        else:
            val_loss = None
            val_acc = None

        # Testing loop
        if (epoch+1) % test_every_x_steps == 0:
            classifier.eval()
            test_losses = []
            test_preds = []
            test_labels = []
            pbar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch+1}/{n_epochs}')
            for batch in pbar:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.no_grad():
                    features = get_features(encoder, inputs, encoder_target_dim, device=device)
                    outputs = classifier(features)

                loss = criterion(outputs, labels)
                test_losses.append(loss.item())
                pbar.set_postfix({"Test Loss": loss.item()})

                if train_dataset.is_multilabel():
                    predicted = (torch.sigmoid(outputs) > 0.5).int()
                    test_preds.extend(predicted.flatten().cpu().numpy())
                    test_labels.extend(labels.flatten().cpu().numpy())
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    test_preds.extend(predicted.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
            
            test_loss = sum(test_losses) / len(test_losses)
            test_acc = 100.0 * (np.array(test_preds) == np.array(test_labels)).sum() / len(test_labels)
            tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
            
            # Early stopping
            if early_stopping_based_on_test:
                recent_accs = [test_record["test_accuracy"] for test_record in history[-5:]]
                recent_accs.append(test_acc)
                # If no improvement in last 6 epochs, stop training
                if (len(recent_accs) >= 6) and (max(recent_accs) - min(recent_accs) < 0.05):
                    print("Early stopping triggered. No improvement in test accuracy.")
                    history.append({
                        "epoch": epoch + 1, 
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "test_loss": test_loss,
                        "test_accuracy": test_acc
                    })
                    save_checkpoint(chkpt_filepath, classifier, optimizer, epoch + 1, history, hyperparams, early_stopped=True)
                    break
        else:
            test_loss = None
            test_acc = None

        # Save checkpoint
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })

        save_checkpoint(chkpt_filepath, classifier, optimizer, epoch + 1, history, hyperparams, early_stopped=False)
