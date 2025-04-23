
"""
Analysis script for arithmetic transformer model.
Performs detailed analysis of model performance on different input types.
"""

import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import ArithmeticDataset, create_data_loaders
from src.model.transformer import make_model, Batch, greedy_decode
from src.utils.helpers import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Arithmetic Transformer Model Performance')
    parser.add_argument('--model_path', type=str, default='models/best_arithmetic_transformer.pt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for analysis')
    parser.add_argument('--test_size', type=int, default=5000, help='Number of test examples')
    parser.add_argument('--max_digits_num1', type=int, default=3, 
                        help='Maximum digits in first number')
    parser.add_argument('--max_digits_num2', type=int, default=3, 
                        help='Maximum digits in second number')
    parser.add_argument('--operations', type=str, default='+,-', 
                        help='Operations to test (comma-separated)')
    parser.add_argument('--error_examples', type=int, default=10,
                        help='Number of error examples to display')
    return parser.parse_args()

def analyze_errors(model, data_loader, dataset, max_examples=10):
    """Analyze errors made by the model."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for src, tgt, input_strs, target_strs in tqdm(data_loader, desc="Finding errors"):
            batch = Batch(src, tgt, pad=0)
            batch_size = src.size(0)
            
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
                
                pred_str = ''.join(pred_tokens)
                
                # If prediction is wrong, add to errors
                if pred_str != target_strs[i]:
                    # Extract data for analysis
                    input_str = input_strs[i]
                    
                    # Check for carry operations in addition
                    has_carry = False
                    has_borrow = False
                    
                    if '+' in input_str:
                        num1, num2 = input_str.split('+')
                        # Check for carry operations
                        num1 = num1.zfill(len(num2))
                        num2 = num2.zfill(len(num1))
                        has_carry = any(int(num1[-(j+1)]) + int(num2[-(j+1)]) >= 10 
                                       for j in range(min(len(num1), len(num2))))
                    elif '-' in input_str:
                        num1, num2 = input_str.split('-')
                        # Check for borrow operations
                        num1 = num1.zfill(len(num2))
                        num2 = num2.zfill(len(num1))
                        has_borrow = any(int(num1[-(j+1)]) < int(num2[-(j+1)]) 
                                        for j in range(min(len(num1), len(num2))))
                    
                    errors.append({
                        'input': input_str,
                        'target': target_strs[i],
                        'prediction': pred_str,
                        'input_length': len(input_str),
                        'has_carry': has_carry,
                        'has_borrow': has_borrow,
                        'error_type': 'wrong_value' if len(pred_str) == len(target_strs[i]) else 'wrong_length'
                    })
                
                # Limit number of examples
                if len(errors) >= max_examples:
                    break
            
            if len(errors) >= max_examples:
                break
    
    return errors

def analyze_by_input_length(model, data_loader, dataset):
    """Analyze performance by input length."""
    model.eval()
    length_metrics = {}
    
    with torch.no_grad():
        for src, tgt, input_strs, target_strs in tqdm(data_loader, desc="Analyzing by length"):
            batch = Batch(src, tgt, pad=0)
            batch_size = src.size(0)
            
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
                
                pred_str = ''.join(pred_tokens)
                
                # Get input length
                input_len = len(input_strs[i])
                
                # Update metrics for this length
                if input_len not in length_metrics:
                    length_metrics[input_len] = {'correct': 0, 'total': 0}
                
                length_metrics[input_len]['total'] += 1
                if pred_str == target_strs[i]:
                    length_metrics[input_len]['correct'] += 1
    
    # Calculate accuracy by length
    for length, metrics in length_metrics.items():
        metrics['accuracy'] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
    
    return length_metrics

def analyze_by_operation(model, data_loader, dataset):
    """Analyze performance by operation (+ or -)."""
    model.eval()
    operation_metrics = {'+': {'correct': 0, 'total': 0, 'with_carry': {'correct': 0, 'total': 0}, 
                              'without_carry': {'correct': 0, 'total': 0}},
                         '-': {'correct': 0, 'total': 0, 'with_borrow': {'correct': 0, 'total': 0}, 
                              'without_borrow': {'correct': 0, 'total': 0}}}
    
    with torch.no_grad():
        for src, tgt, input_strs, target_strs in tqdm(data_loader, desc="Analyzing by operation"):
            batch = Batch(src, tgt, pad=0)
            batch_size = src.size(0)
            
            for i in range(batch_size):
                # Determine operation
                input_str = input_strs[i]
                operation = '+' if '+' in input_str else '-'
                
                # Check for carry/borrow
                has_carry = False
                has_borrow = False
                
                if operation == '+':
                    num1, num2 = input_str.split('+')
                    # Check for carry operations
                    num1 = num1.zfill(len(num2))
                    num2 = num2.zfill(len(num1))
                    has_carry = any(int(num1[-(j+1)]) + int(num2[-(j+1)]) >= 10 
                                   for j in range(min(len(num1), len(num2))))
                else:  # operation == '-'
                    num1, num2 = input_str.split('-')
                    # Check for borrow operations
                    num1 = num1.zfill(len(num2))
                    num2 = num2.zfill(len(num1))
                    has_borrow = any(int(num1[-(j+1)]) < int(num2[-(j+1)]) 
                                    for j in range(min(len(num1), len(num2))))
                
                # Greedy decode
                model_out = greedy_decode(model, src[i:i+1], batch.src_mask[i:i+1], 
                                         max_len=20, start_symbol=1)
                
                # Convert to tokens
                pred_tokens = []
                for j in range(1, model_out.size(1)):
                    idx = model_out[0, j].item()
                    if idx == 2:  # <eos>
                        break
                    pred_tokens.append(dataset.idx_to_char[idx])
                
                pred_str = ''.join(pred_tokens)
                is_correct = pred_str == target_strs[i]
                
                # Update metrics for this operation
                operation_metrics[operation]['total'] += 1
                if is_correct:
                    operation_metrics[operation]['correct'] += 1
                
                # Update carry/borrow metrics
                if operation == '+':
                    key = 'with_carry' if has_carry else 'without_carry'
                    operation_metrics[operation][key]['total'] += 1
                    if is_correct:
                        operation_metrics[operation][key]['correct'] += 1
                else:  # operation == '-'
                    key = 'with_borrow' if has_borrow else 'without_borrow'
                    operation_metrics[operation][key]['total'] += 1
                    if is_correct:
                        operation_metrics[operation][key]['correct'] += 1
    
    # Calculate accuracy by operation
    for op, metrics in operation_metrics.items():
        metrics['accuracy'] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
        
        # Calculate accuracy for carry/borrow sub-metrics
        if op == '+':
            for key in ['with_carry', 'without_carry']:
                sub_metrics = metrics[key]
                sub_metrics['accuracy'] = sub_metrics['correct'] / sub_metrics['total'] if sub_metrics['total'] > 0 else 0
        else:  # op == '-'
            for key in ['with_borrow', 'without_borrow']:
                sub_metrics = metrics[key]
                sub_metrics['accuracy'] = sub_metrics['correct'] / sub_metrics['total'] if sub_metrics['total'] > 0 else 0
    
    return operation_metrics

def visualize_results(length_metrics, operation_metrics):
    """Visualize analysis results."""
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # 1. Plot accuracy by input length
    lengths = sorted(length_metrics.keys())
    accuracies = [length_metrics[l]['accuracy'] for l in lengths]
    totals = [length_metrics[l]['total'] for l in lengths]
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=lengths, y=accuracies)
    
    # Add data labels on bars
    for i, (acc, total) in enumerate(zip(accuracies, totals)):
        ax.annotate(f'{acc:.2f}\nn={total}', 
                   xy=(i, acc), 
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center', va='bottom')
    
    plt.xlabel('Input Length (characters)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Input Length')
    plt.tight_layout()
    plt.savefig('results/accuracy_by_length.png')
    plt.close()
    
    # 2. Plot accuracy by operation
    operations = list(operation_metrics.keys())
    op_accuracies = [operation_metrics[op]['accuracy'] for op in operations]
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=operations, y=op_accuracies)
    
    # Add data labels on bars
    for i, (op, acc) in enumerate(zip(operations, op_accuracies)):
        total = operation_metrics[op]['total']
        ax.annotate(f'{acc:.2f}\nn={total}', 
                   xy=(i, acc), 
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center', va='bottom')
    
    plt.xlabel('Operation')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Operation')
    plt.tight_layout()
    plt.savefig('results/accuracy_by_operation.png')
    plt.close()
    
    # 3. Plot accuracy by carry/borrow
    plt.figure(figsize=(12, 6))
    
    # Create data for grouped bar chart
    categories = []
    accuracies = []
    group_labels = []
    
    # Addition with/without carry
    categories.extend(['With Carry', 'Without Carry'])
    accuracies.extend([
        operation_metrics['+']['with_carry']['accuracy'],
        operation_metrics['+']['without_carry']['accuracy']
    ])
    group_labels.extend(['Addition', 'Addition'])
    
    # Subtraction with/without borrow
    categories.extend(['With Borrow', 'Without Borrow'])
    accuracies.extend([
        operation_metrics['-']['with_borrow']['accuracy'],
        operation_metrics['-']['without_borrow']['accuracy']
    ])
    group_labels.extend(['Subtraction', 'Subtraction'])
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        'Category': categories,
        'Accuracy': accuracies,
        'Operation': group_labels
    })
    
    # Plot grouped bar chart
    ax = sns.barplot(x='Category', y='Accuracy', hue='Operation', data=df)
    
    # Add data labels
    for i, p in enumerate(ax.patches):
        if i < 2:  # Addition metrics
            total = operation_metrics['+']['with_carry' if i == 0 else 'without_carry']['total']
        else:  # Subtraction metrics
            total = operation_metrics['-']['with_borrow' if i == 2 else 'without_borrow']['total']
        
        ax.annotate(f'{p.get_height():.2f}\nn={total}', 
                   xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center', va='bottom')
    
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Operation and Complexity')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('results/accuracy_by_operation_complexity.png')
    plt.close()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse operations
    operations = args.operations.split(',')
    
    # Create test data loader
    _, _, test_loader, _, vocab_size, dataset = create_data_loaders(
        batch_size=args.batch_size,
        train_size=1,  # Not used for analysis
        val_size=1,    # Not used for analysis
        test_size=args.test_size,
        max_digits_num1=args.max_digits_num1,
        max_digits_num2=args.max_digits_num2,
        operations=operations
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract model hyperparameters from checkpoint, or use defaults
    model_config = checkpoint.get('model_config', {
        'N': 3,
        'd_model': 128,
        'd_ff': 512,
        'h': 8,
        'dropout': 0.1
    })
    
    # Create model with same architecture
    model = make_model(
        vocab_size, vocab_size,
        N=model_config.get('N', 3),
        d_model=model_config.get('d_model', 128),
        d_ff=model_config.get('d_ff', 512),
        h=model_config.get('h', 8),
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 1. Find and analyze errors
    print("Analyzing model errors...")
    errors = analyze_errors(model, test_loader, dataset, max_examples=args.error_examples)
    
    print(f"\nFound {len(errors)} errors. Examples:")
    for i, error in enumerate(errors):
        print(f"Error {i+1}:")
        print(f"  Input: {error['input']}")
        print(f"  Target: {error['target']}")
        print(f"  Prediction: {error['prediction']}")
        print(f"  Input Length: {error['input_length']}")
        print(f"  Has Carry: {error['has_carry']}")
        print(f"  Has Borrow: {error['has_borrow']}")
        print(f"  Error Type: {error['error_type']}")
        print()
    
    # Save errors to CSV for further analysis
    errors_df = pd.DataFrame(errors)
    os.makedirs('results', exist_ok=True)
    errors_df.to_csv('results/model_errors.csv', index=False)
    print("Saved error examples to results/model_errors.csv")
    
    # 2. Analyze by input length
    print("\nAnalyzing performance by input length...")
    length_metrics = analyze_by_input_length(model, test_loader, dataset)
    
    print("Accuracy by input length:")
    for length in sorted(length_metrics.keys()):
        metrics = length_metrics[length]
        print(f"  Length {length}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    # 3. Analyze by operation
    print("\nAnalyzing performance by operation...")
    operation_metrics = analyze_by_operation(model, test_loader, dataset)
    
    print("Accuracy by operation:")
    for op, metrics in operation_metrics.items():
        print(f"  {op}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        if op == '+':
            carry_metrics = metrics['with_carry']
            no_carry_metrics = metrics['without_carry']
            print(f"    With carry: {carry_metrics['accuracy']:.4f} ({carry_metrics['correct']}/{carry_metrics['total']})")
            print(f"    Without carry: {no_carry_metrics['accuracy']:.4f} ({no_carry_metrics['correct']}/{no_carry_metrics['total']})")
        else:
            borrow_metrics = metrics['with_borrow']
            no_borrow_metrics = metrics['without_borrow']
            print(f"    With borrow: {borrow_metrics['accuracy']:.4f} ({borrow_metrics['correct']}/{borrow_metrics['total']})")
            print(f"    Without borrow: {no_borrow_metrics['accuracy']:.4f} ({no_borrow_metrics['correct']}/{no_borrow_metrics['total']})")
    
    # 4. Visualize results
    print("\nGenerating visualizations...")
    visualize_results(length_metrics, operation_metrics)
    print("Visualizations saved to results/ directory")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())