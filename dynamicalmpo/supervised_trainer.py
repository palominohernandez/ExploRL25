import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import traceback 



try:
    from datasets import SmilesDataset, collate_fn, load_smiles_data 
    from model_definition import SmilesRNN
    from vocabulary import SmilesVocabulary, load_vocabulary, save_vocabulary 
    from utils import PAD_token, load_checkpoint, save_checkpoint, create_vocabulary_from_data 
    from agent import SmilesGeneratorAgent
    COMPONENT_IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Error importing components in supervised_trainer.py: {e}")
    print("Please ensure datasets.py, model_definition.py, vocabulary.py, utils.py, agent.py are accessible.")
    COMPONENT_IMPORTS_SUCCESS = False

class ModelFactory:
    _models = {
        'SmilesRNN' : SmilesRNN}

    @staticmethod
    def create_model(model_name: str, **kwargs):
        if model_name not in ModelFactory._models:
            raise ValueError(f"Unknown model name: {model_name}")
        return ModelFactory._models[model_name](**kwargs)

def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    """Performs a single training epoch."""
    if not COMPONENT_IMPORTS_SUCCESS: raise RuntimeError("Component imports failed.") 
    model.train()
    total_loss = 0
    total_correct = 0
    total_chars = 0
    num_batches = 0
    dataset_size = len(dataloader.dataset) if dataloader.dataset else 0 

    for batch_data in dataloader:
        try:
            inputs, targets, lengths = batch_data
            if inputs is None or targets is None or lengths is None: 
                continue
        except ValueError:
            continue

        if not isinstance(lengths, torch.Tensor) or not torch.all(lengths > 0):
            continue 

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        try:
            policy_logits, _ = model(inputs, lengths) 
        except Exception as model_err:
             print(f"Error during model forward pass in train_epoch: {model_err}")
             continue 

        vocab_size = getattr(model, 'vocab_size', None) or getattr(model, 'output_size', None) 
        if vocab_size is None: raise AttributeError("Model must have vocab_size or output_size attribute.")

        output_flat = policy_logits.view(-1, vocab_size)
        target_flat = targets.view(-1)


        loss = criterion(output_flat, target_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0) 

        with torch.no_grad(): 
             _, predicted = torch.max(policy_logits.data, 2) 
             mask = (targets != PAD_token) 
             total_correct += (predicted[mask] == targets[mask]).sum().item()
             total_chars += mask.sum().item()
        num_batches += 1

    if num_batches == 0: return 0.0, 0.0 

    avg_loss = total_loss / dataset_size if dataset_size > 0 else 0
    accuracy = total_correct / total_chars if total_chars > 0 else 0
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a dataset."""
    if not COMPONENT_IMPORTS_SUCCESS: raise RuntimeError("Component imports failed.") 
    model.eval() 
    total_loss = 0
    total_correct = 0
    total_chars = 0
    num_batches = 0
    dataset_size = len(dataloader.dataset) if dataloader.dataset else 0 

    with torch.no_grad(): 
        for batch_data in dataloader:
            try:
                inputs, targets, lengths = batch_data
                if inputs is None or targets is None or lengths is None:
                     continue
            except ValueError: continue 

            if not isinstance(lengths, torch.Tensor) or not torch.all(lengths > 0):
                continue 

            inputs, targets = inputs.to(device), targets.to(device)

            try:
                policy_logits, _ = model(inputs, lengths)
            except Exception as model_err:
                 print(f"Error during model forward pass in evaluate: {model_err}")
                 continue 

            vocab_size = getattr(model, 'vocab_size', None) or getattr(model, 'output_size', None)
            if vocab_size is None: raise AttributeError("Model must have vocab_size or output_size attribute.")

            output_flat = policy_logits.view(-1, vocab_size)
            target_flat = targets.view(-1)

            loss = criterion(output_flat, target_flat)
            total_loss += loss.item() * inputs.size(0) 

            _, predicted = torch.max(policy_logits.data, 2)
            mask = (targets != PAD_token)
            total_correct += (predicted[mask] == targets[mask]).sum().item()
            total_chars += mask.sum().item()
            num_batches += 1

    if num_batches == 0: return 0.0, 0.0

    avg_loss = total_loss / dataset_size if dataset_size > 0 else 0
    accuracy = total_correct / total_chars if total_chars > 0 else 0
    return avg_loss, accuracy


def train_supervised(args, device):
    """Handles the supervised pre-training process."""
    if not COMPONENT_IMPORTS_SUCCESS: print("Cannot run supervised training due to import errors."); return 

    print("\n--- Running Supervised Pre-training ---")

    all_smiles = load_smiles_data(args.data_file)
    if all_smiles is None: print("Failed to load data."); return
    subset_fraction = getattr(args, 'subset_fraction', 1.0)
    if 0.0 < subset_fraction < 1.0:
        subset_size = int(len(all_smiles) * subset_fraction)
        print(f"Using random subset: {subset_size}/{len(all_smiles)}")
        all_smiles = list(np.random.choice(all_smiles, subset_size, replace=False))
    else:
        print(f"Using all {len(all_smiles)} samples.")

    vocab = load_vocabulary(args.vocab_path)
    if vocab is None:
        print("Vocabulary not found, creating from data...")
        all_smiles_for_vocab = load_smiles_data(args.data_file)
        if all_smiles_for_vocab is None: print("Failed to load data for vocab creation."); exit(1)
        vocab = create_vocabulary_from_data(all_smiles_for_vocab)
        if vocab: save_vocabulary(vocab, args.vocab_path)
        else: print("Failed to create vocabulary."); exit(1)
    if vocab is None: print("Failed to load/create vocabulary."); exit(1) 


    val_size = getattr(args, 'val_size', 0.15)
    test_size = getattr(args, 'test_size', 0.15)
    total_val_test = val_size + test_size
    if not (0 <= total_val_test < 1.0):
        print("Error: Sum of validation and test sizes must be >= 0 and < 1.0"); return

    if total_val_test > 0:
        try:
            train_smiles, temp_smiles = train_test_split(all_smiles, test_size=total_val_test, random_state=args.seed)
            if val_size == 0: val_smiles, test_smiles = [], temp_smiles
            elif test_size == 0: val_smiles, test_smiles = temp_smiles, []
            else:
                test_fraction = test_size / total_val_test
                val_smiles, test_smiles = train_test_split(temp_smiles, test_size=test_fraction, random_state=args.seed)
        except Exception as e: print(f"Error during data split: {e}"); return
    else:
        train_smiles, val_smiles, test_smiles = all_smiles, [], []
    print(f"Data split: Train={len(train_smiles)}, Validation={len(val_smiles)}, Test={len(test_smiles)}")



    try:
        train_dataset = SmilesDataset(train_smiles, vocab)
        val_dataset = SmilesDataset(val_smiles, vocab) if val_smiles else None
        test_dataset = SmilesDataset(test_smiles, vocab) if test_smiles else None

        batch_size = getattr(args, 'batch_size', 512)
        num_workers = getattr(args, 'num_workers', 0)
        pin_memory = False if str(device) == 'mps' else (True if device != torch.device('cpu') else False)


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory) if test_dataset else None
    except Exception as e: print(f"Error creating Datasets/DataLoaders: {e}"); return


    model = ModelFactory.create_model(
        model_name=args.model,
        vocab_size=vocab.n_chars, 
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
    print("\nModel Architecture:")
    print(model)
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    start_epoch = 1
    best_val_loss = float('inf')
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} 
    supervised_checkpoint_dir = getattr(args, 'supervised_checkpoint_dir', 'checkpoints/supervised')
    os.makedirs(supervised_checkpoint_dir, exist_ok=True)
    supervised_ckpt_path = os.path.join(supervised_checkpoint_dir, "supervised_checkpoint.pth.tar")

    if getattr(args, 'resume_supervised', False):
        if os.path.isfile(supervised_ckpt_path):
            print(f"Resuming supervised training from {supervised_ckpt_path}")
            checkpoint = load_checkpoint(supervised_ckpt_path, model=model, optimizer=optimizer, device=device)
            if checkpoint:
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))

                loaded_history = checkpoint.get('history', None)
                if loaded_history and isinstance(loaded_history, dict): 
                     history = loaded_history
                     if 'epoch' not in history: history['epoch'] = list(range(1, start_epoch))

                print(f"Resumed from epoch {start_epoch}. Best validation loss: {best_val_loss:.4f}")
            else: print(f"Warning: Failed to load checkpoint file {supervised_ckpt_path}. Starting from scratch.")
        else: print(f"Warning: Supervised checkpoint {supervised_ckpt_path} not found for resuming. Starting from scratch.")


    epochs_no_improve = 0
    start_time_total = time.time()
    print("\n--- Starting Supervised Training Loop ---")
    patience = getattr(args, 'patience', 5) 
    max_epochs = getattr(args, 'epochs', 25) 

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip) 

        val_loss, val_acc = float('inf'), 0.0
        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            epoch_duration = time.time() - epoch_start_time

            history['epoch'].append(epoch); 
            history['train_loss'].append(float(train_loss)); history['train_acc'].append(float(train_acc))
            history['val_loss'].append(float(val_loss)); history['val_acc'].append(float(val_acc))
            print(f"Epoch {epoch}/{max_epochs} | T: {epoch_duration:.2f}s | Train L: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val L: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_no_improve = 0
                state = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss, 'vocab_size': vocab.n_chars,
                    'history': history, 'args': vars(args) 
                }
                save_checkpoint(state, filename=supervised_ckpt_path) 
                torch.save(model.state_dict(), args.supervised_model_path)
                print(f"   Val loss improved to {best_val_loss:.4f}. Saved checkpoint and best model to '{args.supervised_model_path}'.")
            else:
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement.")
                    break 

        else:
            epoch_duration = time.time() - epoch_start_time
            history['epoch'].append(epoch);
            history['train_loss'].append(float(train_loss)); history['train_acc'].append(float(train_acc))
            history['val_loss'].append(float('nan')); history['val_acc'].append(float('nan'))
            print(f"Epoch {epoch}/{max_epochs} | T: {epoch_duration:.2f}s | Train L: {train_loss:.4f} | Train Acc: {train_acc:.4f} | (No Validation)")

            if epoch % 5 == 0 or epoch == max_epochs: 
                torch.save(model.state_dict(), args.supervised_model_path)
                print(f"   Saved model state at epoch {epoch} to {args.supervised_model_path}")


        print(f"--- [Epoch {epoch}] Generating Samples ---")
        model.eval() 
        try:
            temp_agent = SmilesGeneratorAgent(model, vocab, device)

            sampled_smiles, _, _, _, _ = temp_agent.generate_trajectories(
                batch_size=5,
                max_len=getattr(args, 'max_gen_len', 120), 
            )

            if sampled_smiles:
                for i, smi in enumerate(sampled_smiles):
                    print(f"  Sample {i+1}: {smi}")
            else:
                print("  Failed to generate samples.")

        except AttributeError as attr_err:
             print(f"  Error during sampling setup (AttributeError): {attr_err}")
             print("  Ensure Vocabulary class has 'char_to_int', 'int_to_char', 'n_chars' attributes.")
        except Exception as e:
            print(f"  Error during sampling: {e}")
            traceback.print_exc() 
        finally:
             model.train() 
        print("-" * 30)


    total_training_time = time.time() - start_time_total
    print(f"\nSupervised training finished. Total time: {total_training_time:.2f}s")

    if test_loader:
        print("\n--- Loading best model for final test evaluation ---")
        if os.path.exists(args.supervised_model_path):
            try:

                test_model = SmilesRNN(vocab.n_chars, args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout).to(device)
                load_checkpoint(args.supervised_model_path, model=test_model, device=device)
                test_loss, test_acc = evaluate(test_model, test_loader, criterion, device)
                print(f"Test Set Performance -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
                history['test_loss'] = test_loss
                history['test_acc'] = test_acc
            except Exception as e:
                print(f"Could not load or evaluate best model from {args.supervised_model_path}: {e}")
        else:
            print(f"Best supervised model file not found at {args.supervised_model_path}. Skipping test evaluation.")


    results_subdir = os.path.join(args.results_dir, 'supervised_training')
    os.makedirs(results_subdir, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(args.supervised_model_path))[0]
    history_save_path = os.path.join(results_subdir, run_name + "_history.json")
    plot_save_path = os.path.join(results_subdir, run_name + "_curves.png")

    try:
        serializable_history = {}
        for key, value in history.items():
             if isinstance(value, list) and value and isinstance(value[0], (torch.Tensor, np.generic)):
                  serializable_history[key] = [v.item() if isinstance(v, torch.Tensor) else float(v) for v in value]
             else:
                  serializable_history[key] = value 

        with open(history_save_path, 'w') as f: json.dump(serializable_history, f, indent=4)
        print(f"Saved supervised training history to {history_save_path}")
    except Exception as e:
        print(f"Error saving history: {e}")

    try:
        epochs_ran = history.get('epoch', list(range(1, len(history['train_loss']) + 1))) 
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_ran, history['train_loss'], label='Train Loss')
        if 'val_loss' in history and any(not np.isnan(v) for v in history['val_loss']):
             plt.plot(epochs_ran, history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Supervised Loss'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(epochs_ran, history['train_acc'], label='Train Accuracy')
        if 'val_acc' in history and any(not np.isnan(v) for v in history['val_acc']):
             plt.plot(epochs_ran, history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Supervised Accuracy'); plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig(plot_save_path)
        print(f"Saved training plot to {plot_save_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot supervised training curves: {e}")

