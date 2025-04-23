
"""
Evaluation script for arithmetic transformer model.
"""

import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import ArithmeticDataset, create_data_loaders
from src.model.transformer import make_model, Batch
from src.utils.helpers import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Arithmetic Transformer Model')
    parser.add_argument('--model_path', type=str, default='models/best_arithmetic_transformer.pt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--test_size', type=int, default=5000, help='Number of test examples')
    parser.add_argument('--gen_size', type=int, default=5000, help='Number of generalization examples')
    parser.add_argument('--max_digits_num1', type=int, default=3, 
                        help='Maximum digits in first number for test set')
    parser.add_argument('--max_digits_num2', type=int, default=3, 
                        help='Maximum digits in second number for test set')
    parser.add_argument('--gen_digits', type=int, default=5, 
                        help='Number of digits for generalization test')
    parser.add_argument('--operations', type=str, default='+,-', 
                        help='Operations to test (comma-separated)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse operations
    operations = args.operations.split(',')
    
    # Create test and generalization data loaders
    _, _, test_loader, gen_loader, vocab_size, dataset = create_data_loaders(
        batch_size=args.batch_size,
        train_size=1,  # Not used for evaluation
        val_size=1,    # Not used for evaluation
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
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, dataset)
    print(f"Test Metrics:")
    print(f"  Exact Match Accuracy: {test_metrics['exact_match_accuracy']:.4f}")
    print(f"  Character-Level Accuracy: {test_metrics['char_level_accuracy']:.4f}")
    print(f"  Perplexity: {test_metrics['perplexity']:.4f}")
    
    # Evaluate on generalization set
    print("\nEvaluating on generalization set (longer numbers)...")
    gen_metrics = evaluate_model(model, gen_loader, dataset)
    print(f"Generalization Metrics:")
    print(f"  Exact Match Accuracy: {gen_metrics['exact_match_accuracy']:.4f}")
    print(f"  Character-Level Accuracy: {gen_metrics['char_level_accuracy']:.4f}")
    print(f"  Perplexity: {gen_metrics['perplexity']:.4f}")
    
    # Plot comparison
    metrics = ['exact_match_accuracy', 'char_level_accuracy']
    labels = ['Exact Match Accuracy', 'Character-Level Accuracy']
    
    plt.figure(figsize=(10, 6))
    x = range(len(metrics))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], [test_metrics[m] for m in metrics], width, label='Test Set')
    plt.bar([i + width/2 for i in x], [gen_metrics[m] for m in metrics], width, label='Generalization Set')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/evaluation_comparison.png')
    plt.show()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())