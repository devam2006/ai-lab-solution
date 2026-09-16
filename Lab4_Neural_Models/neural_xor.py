import torch
import torch.nn as nn
import torch.optim as optim
import math

# --- XOR Dataset ---
# Inputs: x1, x2
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

# XOR Labels for binary classification
y_binary = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Labels for 3-class classification
# 0: both inactive, 1: disagree, 2: both active
y_multi = torch.tensor([0, 1, 1, 2])

class BinaryXORNet(nn.Module):
    def __init__(self, activation_name="tanh", init_zero=False):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        
        if activation_name == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation_name == "tanh":
            self.act = nn.Tanh()
        elif activation_name == "relu":
            self.act = nn.ReLU()
            
        if init_zero:
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out) # Logits
        return out

class MultiClassXORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(2, 3) # 3 outputs for 3 classes

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out) # Logits
        return out

def train_binary(name, model, epochs=2000, lr=0.1, print_freq=0):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    print(f"\n--- {name} ---")
    initial_loss = None
    
    # Store early gradient norm
    early_grad_norm = 0.0
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y_binary)
        
        if epoch == 0:
            initial_loss = loss.item()
            
        loss.backward()
        
        if epoch == 1:
            # Capture Euclidean norm of first layer gradient at step 1
            early_grad_norm = torch.norm(model.fc1.weight.grad).item()
            
        if print_freq > 0 and (epoch == 0 or epoch == epochs):
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            if epoch == 1:
                print(f"Gradient of fc1 weights:\n{model.fc1.weight.grad}")
                
        optimizer.step()
        
    final_loss = loss.item()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
    correct = (preds == y_binary).sum().item()
    
    print(f"Initial Loss: {initial_loss:.4f} | Final Loss: {final_loss:.4f}")
    print(f"Final Probabilities:\n{probs.squeeze().numpy()}")
    print(f"Thresholded Predictions:\n{preds.squeeze().numpy()}")
    print(f"Correct: {correct}/4 | Early Grad Norm: {early_grad_norm:.6f}")
    return final_loss, correct, early_grad_norm

def train_multiclass():
    model = MultiClassXORNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    print("\n--- Task 5: Three-Class Extension ---")
    
    for epoch in range(2000):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y_multi)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
    print(f"Final Loss: {loss.item():.4f}")
    print(f"Logits:\n{logits.numpy()}")
    print(f"Probabilities:\n{probs.numpy()}")
    print(f"Predictions: {preds.numpy()}")
    
    # Check sum to 1
    sum_probs = probs[0].sum().item()
    print(f"Sum of probabilities for example 0: {sum_probs:.6f}")

if __name__ == "__main__":
    # Task 4 Part A & B: Basic Learning & Backprop check
    basic_net = BinaryXORNet(activation_name="tanh")
    train_binary("Part A & B: Basic Learning Check (Tanh)", basic_net, epochs=5000, lr=0.5, print_freq=5000)
    print("First layer weights grad after final step:")
    print(basic_net.fc1.weight.grad)
    
    # Task 4 Part C: Symmetry Experiment
    sym_net = BinaryXORNet(activation_name="tanh", init_zero=True)
    train_binary("Part C: Symmetry Experiment (Zero Init)", sym_net, epochs=1000, lr=0.5)
    
    # Task 4 Part D: Activation Experiment
    acts = ["sigmoid", "tanh", "relu"]
    for act in acts:
        # Use a fixed seed for comparable initialization
        torch.manual_seed(42)
        net = BinaryXORNet(activation_name=act)
        train_binary(f"Part D: Activation Experiment ({act})", net, epochs=2000, lr=0.5)
        
    # Task 5: Multiclass
    torch.manual_seed(42)
    train_multiclass()
