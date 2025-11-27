"""
The Seven Pillars of Statistical Wisdom in Modern AI/ML
Extended Implementation with Advanced Concepts

Based on Stephen M. Stigler's "The Seven Pillars of Statistical Wisdom"
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_regression, make_classification, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier, Ridge, LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def section_header(title, subtitle=""):
    print(f"\n{'='*70}")
    print(f"PILLAR {title}")
    if subtitle:
        print(f"{subtitle}")
    print(f"{'='*70}")

def subsection_header(title):
    print(f"\n{'-'*70}")
    print(f"{title}")
    print(f"{'-'*70}")

# ==========================================
# 1. AGGREGATION (The Wisdom of Crowds)
# ==========================================
section_header("1: AGGREGATION", "The Wisdom of Crowds")

print("\n📚 CONCEPT: The counter-intuitive idea that aggregating many weak learners")
print("produces better predictions than any single strong learner.")

# 1A: Classical Random Forest
subsection_header("1A: Random Forest - Bootstrap Aggregating (Bagging)")

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_acc = tree.score(X_test, y_test)

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
forest_acc = forest.score(X_test, y_test)

print(f"Single Decision Tree: {tree_acc:.4f}")
print(f"Random Forest (100 trees): {forest_acc:.4f}")
print(f"Improvement: {(forest_acc - tree_acc)*100:.2f}%")

# 1B: Mixture of Experts Simulation
subsection_header("1B: Mixture of Experts (MoE) - Specialized Aggregation")

print("\n💡 Modern AI Concept: Different 'experts' specialize in different regions")
print("of the input space (used in models like Mixtral, GPT-4)")

# Create data with distinct clusters
X_moe, y_moe = make_blobs(n_samples=600, centers=3, n_features=2, 
                          cluster_std=1.5, random_state=42)
X_train_moe, X_test_moe, y_train_moe, y_test_moe = train_test_split(
    X_moe, y_moe, test_size=0.3, random_state=42
)

# Expert models - each specializes in different regions
expert1 = LogisticRegression(random_state=42)
expert2 = LogisticRegression(random_state=43)
expert3 = LogisticRegression(random_state=44)

# Train experts on different random subsets (simulating specialization)
indices = np.arange(len(X_train_moe))
np.random.shuffle(indices)
split = len(indices) // 3

expert1.fit(X_train_moe[indices[:split]], y_train_moe[indices[:split]])
expert2.fit(X_train_moe[indices[split:2*split]], y_train_moe[indices[split:2*split]])
expert3.fit(X_train_moe[indices[2*split:]], y_train_moe[indices[2*split:]])

# Gating network decides which expert to trust
gating = LogisticRegression(multi_class='multinomial', random_state=42)
# Create pseudo-labels for which expert should handle which region
expert_labels = np.zeros(len(X_train_moe), dtype=int)
expert_labels[indices[:split]] = 0
expert_labels[indices[split:2*split]] = 1
expert_labels[indices[2*split:]] = 2
gating.fit(X_train_moe, expert_labels)

# MoE prediction: weighted combination based on gating network
gate_probs = gating.predict_proba(X_test_moe)
expert_preds = np.array([
    expert1.predict_proba(X_test_moe),
    expert2.predict_proba(X_test_moe),
    expert3.predict_proba(X_test_moe)
])

# Weighted average of expert predictions
moe_proba = np.zeros_like(expert_preds[0])
for i in range(len(X_test_moe)):
    for j in range(3):
        moe_proba[i] += gate_probs[i, j] * expert_preds[j, i]

moe_predictions = np.argmax(moe_proba, axis=1)
moe_acc = accuracy_score(y_test_moe, moe_predictions)

# Compare to single model
single_model = LogisticRegression(random_state=42)
single_model.fit(X_train_moe, y_train_moe)
single_acc = single_model.score(X_test_moe, y_test_moe)

print(f"Single Model Accuracy: {single_acc:.4f}")
print(f"Mixture of Experts Accuracy: {moe_acc:.4f}")
print("→ MoE allows specialized experts to handle different input regions")

# 1C: Dropout as Implicit Ensemble
subsection_header("1C: Dropout - Training an Implicit Ensemble")

print("\n💡 Modern AI Concept: Dropout trains an exponential ensemble of")
print("thinned networks, then averages at test time")

class SimpleDropoutNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Train with dropout
X_drop_train = torch.FloatTensor(X_train)
y_drop_train = torch.LongTensor(y_train)
X_drop_test = torch.FloatTensor(X_test)
y_drop_test = torch.LongTensor(y_test)

net_dropout = SimpleDropoutNet(dropout_rate=0.5)
optimizer = optim.Adam(net_dropout.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Quick training
net_dropout.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = net_dropout(X_drop_train)
    loss = criterion(outputs, y_drop_train)
    loss.backward()
    optimizer.step()

# At test time, dropout is off - we're averaging over all possible subnetworks
net_dropout.eval()
with torch.no_grad():
    test_outputs = net_dropout(X_drop_test)
    dropout_acc = (test_outputs.argmax(dim=1) == y_drop_test).float().mean().item()

print(f"Network with Dropout Accuracy: {dropout_acc:.4f}")
print("→ Each forward pass during training uses a different sub-network")
print("→ At test time, we implicitly ensemble all 2^50 possible sub-networks")


# ==========================================
# 2. INFORMATION (Diminishing Returns)
# ==========================================
section_header("2: INFORMATION", "The Square Root Law")

print("\n📚 CONCEPT: Information scales with √N, not N. Doubling data doesn't")
print("double accuracy - you need 4x more data to halve your error.")

# 2A: Classical Scaling
subsection_header("2A: Classical Data Scaling - The √N Rule")

data_sizes = [50, 100, 200, 400, 800, 1600, 3200]
accuracies = []

X_info, y_info = make_classification(n_samples=5000, n_features=20, 
                                      n_informative=10, random_state=42)

print(f"\n{'Training Size':<15} | {'Accuracy':<10} | {'Gain from 2x data':<20}")
print("-" * 60)

prev_acc = None
for n in data_sizes:
    X_sub, _, y_sub, _ = train_test_split(
        X_info, y_info, train_size=n, stratify=y_info, random_state=42
    )
    
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_sub, y_sub)
    acc = model.score(X_info[3200:], y_info[3200:])
    accuracies.append(acc)
    
    if prev_acc is not None:
        gain = (acc - prev_acc) * 100
        print(f"{n:<15} | {acc:.4f}    | +{gain:.3f}%")
    else:
        print(f"{n:<15} | {acc:.4f}    | baseline")
    prev_acc = acc

print("\n→ Notice diminishing returns: 50→100 gives ~3% gain, 1600→3200 gives ~0.3%")

# 2B: Neural Scaling Laws
subsection_header("2B: Neural Scaling Laws - Modern LLMs")

print("\n💡 Modern AI Concept: LLM performance follows power laws")
print("L(N) = (N_c / N)^α where N is parameters/data/compute")
print("\nSimulated scaling law for loss vs model size:")

# Simulate a scaling law
model_sizes = np.array([1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10])
N_c = 1e10  # Critical scale
alpha = 0.076  # Typical exponent for neural scaling

losses = (N_c / model_sizes) ** alpha

print(f"\n{'Model Size (params)':<20} | {'Loss':<10}")
print("-" * 35)
for size, loss in zip(model_sizes, losses):
    if size >= 1e9:
        size_str = f"{size/1e9:.1f}B"
    elif size >= 1e6:
        size_str = f"{size/1e6:.0f}M"
    else:
        size_str = f"{size:.0f}"
    print(f"{size_str:<20} | {loss:.4f}")

print("\n→ Moving from 1B to 10B parameters (10x) only reduces loss by ~15%")
print("→ This is why GPT-4 required massive compute compared to GPT-3")


# ==========================================
# 3. LIKELIHOOD (Optimization via Loss)
# ==========================================
section_header("3: LIKELIHOOD", "Maximizing Probability of Observed Data")

print("\n📚 CONCEPT: We don't find 'truth'; we find the hypothesis that")
print("maximizes the likelihood of observing our data.")

# 3A: Maximum Likelihood Estimation
subsection_header("3A: Neural Networks as Likelihood Maximizers")

inputs = torch.randn(5, 4, requires_grad=True)
targets = torch.tensor([0, 2, 1, 0, 2])

criterion = nn.CrossEntropyLoss()
outputs = inputs @ torch.randn(4, 3)
loss = criterion(outputs, targets)

print(f"Negative Log Likelihood (Loss): {loss.item():.4f}")
print("→ Training minimizes this, thereby maximizing P(data | model)")

# 3B: Contrastive Learning
subsection_header("3B: Contrastive Learning - Modern Self-Supervision")

print("\n💡 Modern AI Concept: Learn by contrasting positive pairs vs negative pairs")
print("(Used in CLIP, SimCLR, and many foundation models)")

class SimpleContrastiveModel(nn.Module):
    def __init__(self, input_dim=10, embedding_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, embedding_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.encoder(x), dim=1)

# Create synthetic data: original samples and their augmented versions
n_samples = 100
X_contrast = np.random.randn(n_samples, 10)
# Simulate augmentation by adding small noise
X_augmented = X_contrast + np.random.randn(n_samples, 10) * 0.1

X_c_tensor = torch.FloatTensor(X_contrast)
X_aug_tensor = torch.FloatTensor(X_augmented)

model_contrast = SimpleContrastiveModel()
optimizer_c = optim.Adam(model_contrast.parameters(), lr=0.01)

# Contrastive loss: maximize similarity of (x, augmented_x), minimize for others
temperature = 0.5

print("\nTraining with contrastive loss...")
for epoch in range(50):
    optimizer_c.zero_grad()
    
    # Get embeddings
    z_i = model_contrast(X_c_tensor)
    z_j = model_contrast(X_aug_tensor)
    
    # Cosine similarity matrix
    similarity = torch.mm(z_i, z_j.T) / temperature
    
    # Positive pairs are on the diagonal
    labels = torch.arange(n_samples)
    
    # InfoNCE loss (a form of contrastive loss)
    loss_i = F.cross_entropy(similarity, labels)
    loss_j = F.cross_entropy(similarity.T, labels)
    loss_contrast = (loss_i + loss_j) / 2
    
    loss_contrast.backward()
    optimizer_c.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Contrastive Loss = {loss_contrast.item():.4f}")

print("\n→ Model learns to maximize likelihood that augmented pairs match")
print("→ This is the foundation of self-supervised learning in vision & NLP")


# ==========================================
# 4. INTERCOMPARISON (Internal Validation)
# ==========================================
section_header("4: INTERCOMPARISON", "Data Checks Itself")

print("\n📚 CONCEPT: Statistical significance determined internally from data")
print("without external gold standard.")

# 4A: Cross-Validation
subsection_header("4A: K-Fold Cross-Validation")

X_cv, y_cv = make_classification(n_samples=1000, random_state=42)
model_cv = SGDClassifier(random_state=42)

scores = cross_val_score(model_cv, X_cv, y_cv, cv=5)
print(f"5-Fold CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (±{scores.std() * 2:.4f})")
print("→ We split the data into 5 'parallel universes' to check stability")

# 4B: Self-Supervised Learning
subsection_header("4B: Masked Autoencoding - Self-Supervision")

print("\n💡 Modern AI Concept: Hide parts of data, predict from visible parts")
print("(Foundation of BERT, MAE, and modern foundation models)")

# Simulate a simple masked language model task
sequence_length = 20
vocab_size = 100
embedding_dim = 32

class SimpleMaskedPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.predictor = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        return self.predictor(embedded.mean(dim=1))  # Simple average pooling

# Generate random "text" sequences
n_sequences = 200
sequences = torch.randint(0, vocab_size, (n_sequences, sequence_length))

model_mask = SimpleMaskedPredictor(vocab_size, embedding_dim)
optimizer_m = optim.Adam(model_mask.parameters(), lr=0.01)
criterion_m = nn.CrossEntropyLoss()

print("\nTraining masked predictor...")
for epoch in range(30):
    total_loss = 0
    for seq in sequences[:50]:  # Train on subset
        # Randomly mask one position
        mask_pos = np.random.randint(0, sequence_length)
        target_token = seq[mask_pos].item()
        
        # Create masked input
        masked_seq = seq.clone()
        masked_seq[mask_pos] = vocab_size - 1  # Special [MASK] token
        
        optimizer_m.zero_grad()
        output = model_mask(masked_seq.unsqueeze(0))
        loss = criterion_m(output, torch.tensor([target_token]))
        loss.backward()
        optimizer_m.step()
        
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Avg Loss = {total_loss/50:.4f}")

print("\n→ Model learns language structure by predicting masked tokens")
print("→ Pure intercomparison: no external labels needed!")


# ==========================================
# 5. REGRESSION (Regularization)
# ==========================================
section_header("5: REGRESSION", "Regression to the Mean")

print("\n📚 CONCEPT: Extreme observations tend to be followed by less extreme")
print("ones. In ML: prevent overfitting by penalizing extreme parameters.")

# 5A: Classical Regularization
subsection_header("5A: Polynomial Fitting with Ridge Regularization")

X_reg = np.sort(np.random.rand(20, 1) * 10, axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.5, X_reg.shape[0])

poly = PolynomialFeatures(degree=15)
model_overfit = make_pipeline(poly, LinearRegression())
model_overfit.fit(X_reg, y_reg)

model_reg = make_pipeline(poly, Ridge(alpha=1.0))
model_reg.fit(X_reg, y_reg)

X_test_reg = np.linspace(0, 10, 100)[:, np.newaxis]

print(f"Overfit Model |coefficients|: {np.sum(np.abs(model_overfit.steps[1][1].coef_)):.2f}")
print(f"Regularized Model |coefficients|: {np.sum(np.abs(model_reg.steps[1][1].coef_)):.2f}")
print("→ Ridge regression forces coefficients toward zero (the 'mean')")

# 5B: Early Stopping
subsection_header("5B: Early Stopping - Temporal Regularization")

print("\n💡 Modern AI Concept: Stop training before memorizing the training set")

# Create overfitting scenario
X_early, y_early = make_classification(n_samples=200, n_features=20, 
                                        n_informative=5, n_redundant=15, 
                                        random_state=42)
X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(X_early, y_early, 
                                                            test_size=0.3)

# Convert to PyTorch
X_train_t = torch.FloatTensor(X_train_e)
y_train_t = torch.LongTensor(y_train_e)
X_val_t = torch.FloatTensor(X_val_e)
y_val_t = torch.LongTensor(y_val_e)

class OverfitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

net_early = OverfitNet()
optimizer_e = optim.Adam(net_early.parameters(), lr=0.01)
criterion_e = nn.CrossEntropyLoss()

train_losses = []
val_losses = []

print("\n{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12}")
print("-" * 45)

for epoch in range(200):
    # Training
    net_early.train()
    optimizer_e.zero_grad()
    train_out = net_early(X_train_t)
    train_loss = criterion_e(train_out, y_train_t)
    train_loss.backward()
    optimizer_e.step()
    
    # Validation
    net_early.eval()
    with torch.no_grad():
        val_out = net_early(X_val_t)
        val_loss = criterion_e(val_out, y_val_t)
    
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    
    if epoch % 50 == 0:
        print(f"{epoch:<8} | {train_loss.item():<12.4f} | {val_loss.item():<12.4f}")

best_epoch = np.argmin(val_losses)
print(f"\n→ Best validation loss at epoch {best_epoch}")
print(f"→ Training loss kept decreasing: {train_losses[-1]:.4f}")
print(f"→ But validation loss increased after epoch {best_epoch}")
print("→ Early stopping would halt training at the minimum validation loss")

# 5C: Data Augmentation as Regularization
subsection_header("5C: Data Augmentation - Implicit Regularization")

print("\n💡 Modern AI Concept: Adding synthetic variations prevents overfitting")
print("(Essential for vision models, also used in NLP)")

# Simple demonstration with image-like data
X_aug_base = np.random.randn(100, 28*28)
y_aug = np.random.randint(0, 10, 100)

# Without augmentation
X_train_na, X_test_na, y_train_na, y_test_na = train_test_split(
    X_aug_base, y_aug, test_size=0.3
)
model_no_aug = LogisticRegression(max_iter=1000)
model_no_aug.fit(X_train_na, y_train_na)
score_no_aug = model_no_aug.score(X_test_na, y_test_na)

# With augmentation (add rotated/shifted versions)
X_augmented = np.vstack([
    X_train_na,
    X_train_na + np.random.randn(*X_train_na.shape) * 0.1,  # Add noise
    X_train_na * 0.9,  # Scale variation
])
y_augmented = np.hstack([y_train_na, y_train_na, y_train_na])

model_with_aug = LogisticRegression(max_iter=1000)
model_with_aug.fit(X_augmented, y_augmented)
score_with_aug = model_with_aug.score(X_test_na, y_test_na)

print(f"Without augmentation: {score_no_aug:.4f}")
print(f"With augmentation: {score_with_aug:.4f}")
print("→ Augmentation acts as regularization by expanding the training distribution")


# ==========================================
# 6. DESIGN (Randomization)
# ==========================================
section_header("6: DESIGN", "How You Collect Data Matters")

print("\n📚 CONCEPT: The design of data collection matters more than analysis.")
print("Randomization is essential for unbiased learning.")

# 6A: SGD with/without Shuffling
subsection_header("6A: Stochastic Gradient Descent - Why 'Stochastic' Matters")

X_des, y_des = make_classification(n_samples=1000, random_state=42)
sorted_indices = np.argsort(y_des)
X_sorted, y_sorted = X_des[sorted_indices], y_des[sorted_indices]

clf_bad = SGDClassifier(shuffle=False, max_iter=1, warm_start=True, random_state=42)
clf_good = SGDClassifier(shuffle=True, max_iter=1, warm_start=True, random_state=42)

print("\nTraining with sequential batches vs randomized batches...")
for i in range(0, 1000, 100):
    batch_X, batch_y = X_sorted[i:i+100], y_sorted[i:i+100]
    clf_bad.partial_fit(batch_X, batch_y, classes=[0, 1])
    
    rand_idx = np.random.randint(0, 1000, 100)
    clf_good.partial_fit(X_sorted[rand_idx], y_sorted[rand_idx], classes=[0, 1])

print(f"Without shuffling (bad design): {clf_bad.score(X_sorted, y_sorted):.4f}")
print(f"With shuffling (good design): {clf_good.score(X_sorted, y_sorted):.4f}")
print("→ Random sampling is crucial - sequential data creates catastrophic forgetting")

# 6B: Curriculum Learning
subsection_header("6B: Curriculum Learning - Order Matters")

print("\n💡 Modern AI Concept: Learning order affects final performance")
print("(Used in training large models, inspired by human learning)")

# Create easy and hard examples
X_curr, y_curr = make_classification(n_samples=500, n_features=20, 
                                      n_informative=15, random_state=42)

# Define "difficulty" as distance from decision boundary
temp_model = LogisticRegression()
temp_model.fit(X_curr, y_curr)
probas = temp_model.predict_proba(X_curr).max(axis=1)
# Easy examples have high confidence, hard examples have low confidence
difficulty = 1 - probas

# Sort by difficulty
easy_to_hard_idx = np.argsort(difficulty)
hard_to_easy_idx = np.argsort(difficulty)[::-1]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_curr, y_curr, test_size=0.2)

# Recompute difficulty on training set
temp_model2 = LogisticRegression()
temp_model2.fit(X_train_c, y_train_c)
train_probas = temp_model2.predict_proba(X_train_c).max(axis=1)
train_difficulty = 1 - train_probas
train_easy_to_hard = np.argsort(train_difficulty)

# Train with curriculum (easy -> hard)
net_curriculum = nn.Sequential(
    nn.Linear(20, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)
optimizer_curr = optim.Adam(net_curriculum.parameters(), lr=0.01)

# Train with random order
net_random = nn.Sequential(
    nn.Linear(20, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)
optimizer_rand = optim.Adam(net_random.parameters(), lr=0.01)

X_train_tensor = torch.FloatTensor(X_train_c)
y_train_tensor = torch.LongTensor(y_train_c)
X_test_tensor = torch.FloatTensor(X_test_c)
y_test_tensor = torch.LongTensor(y_test_c)

print("\nTraining with curriculum vs random order...")
batch_size = 32
n_epochs = 50

for epoch in range(n_epochs):
    # Curriculum training (easy to hard)
    for i in range(0, len(X_train_c), batch_size):
        if epoch < 25:  # Start with easy examples
            idx = train_easy_to_hard[:(len(train_easy_to_hard)//2)]
        else:
            idx = train_easy_to_hard
        
        batch_idx = idx[np.random.choice(len(idx), min(batch_size, len(idx)))]
        
        optimizer_curr.zero_grad()
        outputs = net_curriculum(X_train_tensor[batch_idx])
        loss = F.cross_entropy(outputs, y_train_tensor[batch_idx])
        loss.backward()
        optimizer_curr.step()
    
    # Random order training
    random_idx = np.random.permutation(len(X_train_c))
    for i in range(0, len(X_train_c), batch_size):
        batch_idx = random_idx[i:i+batch_size]
        
        optimizer_rand.zero_grad()
        outputs = net_random(X_train_tensor[batch_idx])
        loss = F.cross_entropy(outputs, y_train_tensor[batch_idx])
        loss.backward()
        optimizer_rand.step()

# Evaluate
net_curriculum.eval()
net_random.eval()
with torch.no_grad():
    curr_pred = net_curriculum(X_test_tensor).argmax(dim=1)
    rand_pred = net_random(X_test_tensor).argmax(dim=1)
    
    curr_acc = (curr_pred == y_test_tensor).float().mean().item()
    rand_acc = (rand_pred == y_test_tensor).float().mean().item()

print(f"Random order training: {rand_acc:.4f}")
print(f"Curriculum learning (easy→hard): {curr_acc:.4f}")
print("→ Starting with easy examples can improve convergence and generalization")


# ==========================================
# 7. RESIDUAL (Boosting & Architecture)
# ==========================================
section_header("7: RESIDUAL", "Structure in What's Left Over")

print("\n📚 CONCEPT: After fitting a model, examine the residuals (errors).")
print("If residuals have structure, your model missed something important.")

# 7A: Gradient Boosting
subsection_header("7A: Gradient Boosting - Iterative Residual Fitting")

X_res, y_res = make_regression(n_samples=200, noise=10, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_res, y_res)

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                 max_depth=3, random_state=42)
gb.fit(X_train_r, y_train_r)

train_errors = gb.train_score_
print(f"Initial training error: {train_errors[0]:.2f}")
print(f"After 50 iterations: {train_errors[49]:.2f}")
print(f"Final training error: {train_errors[-1]:.2f}")
print(f"Test R² Score: {gb.score(X_test_r, y_test_r):.4f}")
print("\n→ Each tree in the sequence learns from the residuals of previous trees")

# 7B: ResNet Architecture
subsection_header("7B: ResNet - Residual Learning in Architecture")

print("\n💡 Modern AI Concept: Skip connections let layers learn residuals")
print("(Revolutionary architecture that enabled very deep networks)")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Linear(channels, channels)
        self.conv2 = nn.Linear(channels, channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # Add skip connection
        out = F.relu(out)
        return out

class PlainBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = F.relu(out)
        return out

# Create a deep network with and without residual connections
class DeepResNet(nn.Module):
    def __init__(self, n_blocks=10):
        super().__init__()
        self.input_layer = nn.Linear(20, 64)
        self.blocks = nn.ModuleList([ResidualBlock(64) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

class DeepPlainNet(nn.Module):
    def __init__(self, n_blocks=10):
        super().__init__()
        self.input_layer = nn.Linear(20, 64)
        self.blocks = nn.ModuleList([PlainBlock(64) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

# Generate data
X_resnet, y_resnet = make_classification(n_samples=500, n_features=20, random_state=42)
X_train_rn, X_test_rn, y_train_rn, y_test_rn = train_test_split(X_resnet, y_resnet)

X_train_rn_t = torch.FloatTensor(X_train_rn)
y_train_rn_t = torch.LongTensor(y_train_rn)
X_test_rn_t = torch.FloatTensor(X_test_rn)
y_test_rn_t = torch.LongTensor(y_test_rn)

# Train both networks
resnet = DeepResNet(n_blocks=10)
plainnet = DeepPlainNet(n_blocks=10)

optimizer_res = optim.Adam(resnet.parameters(), lr=0.001)
optimizer_plain = optim.Adam(plainnet.parameters(), lr=0.001)

print("\nTraining deep networks with and without residual connections...")
n_epochs_res = 100

resnet_losses = []
plainnet_losses = []

for epoch in range(n_epochs_res):
    # Train ResNet
    resnet.train()
    optimizer_res.zero_grad()
    res_out = resnet(X_train_rn_t)
    res_loss = F.cross_entropy(res_out, y_train_rn_t)
    res_loss.backward()
    optimizer_res.step()
    resnet_losses.append(res_loss.item())
    
    # Train PlainNet
    plainnet.train()
    optimizer_plain.zero_grad()
    plain_out = plainnet(X_train_rn_t)
    plain_loss = F.cross_entropy(plain_out, y_train_rn_t)
    plain_loss.backward()
    optimizer_plain.step()
    plainnet_losses.append(plain_loss.item())

# Final evaluation
resnet.eval()
plainnet.eval()
with torch.no_grad():
    res_pred = resnet(X_test_rn_t).argmax(dim=1)
    plain_pred = plainnet(X_test_rn_t).argmax(dim=1)
    
    res_acc = (res_pred == y_test_rn_t).float().mean().item()
    plain_acc = (plain_pred == y_test_rn_t).float().mean().item()

print(f"\nPlain Network (no skip connections):")
print(f"  Final training loss: {plainnet_losses[-1]:.4f}")
print(f"  Test accuracy: {plain_acc:.4f}")

print(f"\nResNet (with skip connections):")
print(f"  Final training loss: {resnet_losses[-1]:.4f}")
print(f"  Test accuracy: {res_acc:.4f}")

print("\n→ Skip connections allow gradients to flow directly through the network")
print("→ Each layer learns the residual (what to add) rather than the full mapping")
print("→ This enabled networks with 100+ layers (vs ~20 before ResNet)")


# ==========================================
# SYNTHESIS
# ==========================================
section_header("SYNTHESIS", "How the Pillars Work Together")

print("""
🎯 THE BIG PICTURE: Training a Modern LLM Touches All 7 Pillars

1. AGGREGATION: Mixture of Experts, ensemble of attention heads
2. INFORMATION: Scaling laws determine data/compute tradeoffs
3. LIKELIHOOD: Cross-entropy loss minimization (MLE)
4. INTERCOMPARISON: Self-supervised pretraining (next token prediction)
5. REGRESSION: Weight decay, dropout, layer normalization
6. DESIGN: Carefully curated pretraining data, curriculum learning
7. RESIDUAL: Residual connections in every transformer block

These aren't separate techniques - they're different facets of the same
underlying statistical philosophy that Stigler identified in classical
statistics and that we've reinvented in the age of deep learning.
""")

print("\n" + "="*70)
print("IMPLEMENTATION COMPLETE")
print("="*70)
print("\n💡 Next steps:")
print("  1. Create visualizations for each pillar")
print("  2. Add real-world case studies (GPT, AlphaGo, etc.)")
print("  3. Build interactive Jupyter notebook")
print("  4. Compile comprehensive bibliography")
