import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import create_data_loaders
from src.data.utils import Batch
from src.model.transformer import make_model
from src.utils.helpers import SimpleLossCompute

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=3):
    """Train the model with early stopping."""
    train_losses = []
    val_metrics = []
    best_val_acc = 0
    patience_counter = 0
    best_model = None
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for src, tgt, _, _ in train_loader_tqdm:
            batch = Batch(src, tgt, pad=0)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            
            # Calculate loss
            loss = criterion(model.generator(output.contiguous().view(-1, output.size(-1))), 
                            batch.trg_y.contiguous().view(-1))
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            
            # Update progress bar
            train_loader_tqdm.set_postfix(loss=loss.item())
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        val_metric = evaluate_model(model, val_loader, train_loader.dataset)
        val_metrics.append(val_metric)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Metrics: Exact Match Acc={val_metric['exact_match_accuracy']:.4f}, "
              f"Char-Level Acc={val_metric['char_level_accuracy']:.4f}, "
              f"Perplexity={val_metric['perplexity']:.4f}")
        
        # Early stopping
        if val_metric['exact_match_accuracy'] > best_val_acc:
            best_val_acc = val_metric['exact_match_accuracy']
            patience_counter = 0
            best_model = copy.deepcopy(model.state_dict())
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metric,
            }, 'models/best_arithmetic_transformer.pt')
            print(f"Model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, train_losses, val_metrics

def evaluate_model(model, data_loader, dataset):
    """Evaluate model on a given dataset."""
    from src.utils.helpers import greedy_decode
    import math
    
    model.eval()
    exact_match = 0
    char_level_correct = 0
    char_level_total = 0
    
    perplexity_sum = 0
    batch_count = 0
    
    criterion = nn.NLLLoss(reduction='sum', ignore_index=0)
    
    with torch.no_grad():
        for src, tgt, input_strs, target_strs in tqdm(data_loader, desc="Evaluating"):
            batch = Batch(src, tgt, pad=0)
            batch_size = src.size(0)
            
            # For perplexity
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = criterion(model.generator(out.contiguous().view(-1, out.size(-1))),
                             batch.trg_y.contiguous().view(-1))
            
            perplexity = math.exp(loss.item() / batch.ntokens)
            perplexity_sum += perplexity
            batch_count += 1
            
            # For exact match and char-level accuracy
            for i in range(batch_size):
                # Greedy decode
                model_out = greedy_decode(model, src[i:i+1], batch.src_mask[i:i+1], 
                                         max_len=20, start_symbol=1)  # <sos> = 1
                
                # Convert to tokens
                pred_tokens = []
                for j in range(1, model_out.size(1)):  # Skip <sos>
                    idx = model_out[0, j].item()
                    if idx == 2:  # <eos>
                        break
                    pred_tokens.append(dataset.idx_to_char[idx])
                
                # Compare with target
                target_tokens = []
                for j in range(1, len(tgt[i])):  # Skip <sos>
                    idx = tgt[i][j].item()
                    if idx == 0 or idx == 2:  # <pad> or <eos>
                        break
                    target_tokens.append(dataset.idx_to_char[idx])
                
                pred_str = ''.join(pred_tokens)
                target_str = ''.join(target_tokens)
                
                # Exact match
                if pred_str == target_str:
                    exact_match += 1
                
                # Character-level accuracy
                min_len = min(len(pred_str), len(target_str))
                char_level_correct += sum(1 for j in range(min_len) if pred_str[j] == target_str[j])
                char_level_total += max(len(pred_str), len(target_str))
    
    # Calculate metrics
    exact_match_acc = exact_match / len(data_loader.dataset)
    char_level_acc = char_level_correct / char_level_total if char_level_total > 0 else 0
    avg_perplexity = perplexity_sum / batch_count if batch_count > 0 else float('inf')
    
    return {
        'exact_match_accuracy': exact_match_acc,
        'char_level_accuracy': char_level_acc,
        'perplexity': avg_perplexity
    }

def main():
    # Hyperparameters
    batch_size = 64
    train_size = 50000
    val_size = 5000
    test_size = 5000
    max_digits_num1 = 3
    max_digits_num2 = 3
    operations = ['+', '-']
        
    # Model hyperparameters
    N = 3  # Number of layers
    d_model = 128  # Model dimension
    d_ff = 512  # Feed-forward dimension
    h = 8  # Number of attention heads
    dropout = 0.1

    # Training hyperparameters
    epochs = 20
    patience = 3
    lr = 0.0005
    
    # Create