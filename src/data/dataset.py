import random
import torch
from torch.utils.data import Dataset, DataLoader

class ArithmeticDataset(Dataset):
    def __init__(self, num_examples, max_digits_num1, max_digits_num2, operations=['+', '-'], 
                 generalization=False, gen_digits=None):
        self.examples = []
        self.operations = operations
        
        # Create vocabulary
        self.char_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        for i in range(10):
            self.char_to_idx[str(i)] = i + 3
        for op in operations:
            self.char_to_idx[op] = len(self.char_to_idx)
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Generate examples
        if generalization and gen_digits is not None:
            self.generate_examples(num_examples, gen_digits, gen_digits)
        else:
            self.generate_examples(num_examples, max_digits_num1, max_digits_num2)
    
    def generate_examples(self, num_examples, max_digits_num1, max_digits_num2):
        for _ in range(num_examples):
            # Randomly choose number of digits for first and second numbers
            digits_num1 = random.randint(1, max_digits_num1)
            digits_num2 = random.randint(1, max_digits_num2)
            
            # Generate first and second numbers
            num1 = random.randint(10**(digits_num1-1), 10**digits_num1 - 1) if digits_num1 > 1 else random.randint(0, 9)
            num2 = random.randint(10**(digits_num2-1), 10**digits_num2 - 1) if digits_num2 > 1 else random.randint(0, 9)
            
            # Ensure num1 >= num2 for subtraction to avoid negative results
            if '-' in self.operations:
                if random.choice(self.operations) == '-':
                    num1, num2 = max(num1, num2), min(num1, num2)
            
            # Randomly choose operation
            op = random.choice(self.operations)
            
            # Calculate result
            if op == '+':
                result = num1 + num2
            else:  # op == '-'
                result = num1 - num2
            
            # Create input and target strings
            input_str = f"{num1}{op}{num2}"
            target_str = str(result)
            
            # Convert strings to token sequences with <sos> and <eos>
            input_tokens = ['<sos>'] + list(input_str) + ['<eos>']
            target_tokens = ['<sos>'] + list(target_str) + ['<eos>']
            
            # Convert tokens to indices
            input_indices = [self.char_to_idx[token] for token in input_tokens]
            target_indices = [self.char_to_idx[token] for token in target_tokens]
            
            self.examples.append((input_indices, target_indices, input_str, target_str))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    # Sort batch by input sequence length (for PackedSequence)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Unzip batch
    input_indices, target_indices, input_strs, target_strs = zip(*batch)
    
    # Get max sequence lengths
    max_input_len = max(len(seq) for seq in input_indices)
    max_target_len = max(len(seq) for seq in target_indices)
    
    # Pad sequences
    padded_inputs = [seq + [0] * (max_input_len - len(seq)) for seq in input_indices]
    padded_targets = [seq + [0] * (max_target_len - len(seq)) for seq in target_indices]
    
    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.LongTensor(padded_inputs).to(device)
    targets = torch.LongTensor(padded_targets).to(device)
    
    return inputs, targets, input_strs, target_strs

def create_data_loaders(batch_size=32, train_size=10000, val_size=1000, test_size=1000, 
                        max_digits_num1=3, max_digits_num2=3, operations=['+', '-']):
    # Create datasets
    train_dataset = ArithmeticDataset(train_size, max_digits_num1, max_digits_num2, operations)
    val_dataset = ArithmeticDataset(val_size, max_digits_num1, max_digits_num2, operations)
    test_dataset = ArithmeticDataset(test_size, max_digits_num1, max_digits_num2, operations)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            collate_fn=collate_fn)
    
    # Create a generalization dataset with more digits than training
    gen_dataset = ArithmeticDataset(test_size, max_digits_num1+2, max_digits_num2+2, 
                                   operations, generalization=True, gen_digits=max_digits_num1+2)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, 
                           collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, gen_loader, train_dataset.vocab_size, train_dataset