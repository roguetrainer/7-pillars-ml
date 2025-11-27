import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_regression, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier, Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def section_header(title):
    print(f"\n{'='*60}\nPILLAR: {title}\n{'='*60}")

# ==========================================
# 1. AGGREGATION (The Wisdom of Crowds)
# Concept: Ensemble Learning (Random Forest) outperforms individual learners.
# ==========================================
section_header("AGGREGATION")

# Generate noisy non-linear data
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Single Decision Tree (High Variance)
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_acc = tree.score(X_test, y_test)

# Random Forest (Aggregation of Trees)
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
forest_acc = forest.score(X_test, y_test)

print(f"Single Decision Tree Accuracy: {tree_acc:.4f}")
print(f"Random Forest (Aggregation) Accuracy: {forest_acc:.4f}")
print("-> Aggregation reduced variance and improved generalization.")


# ==========================================
# 2. INFORMATION (Diminishing Returns)
# Concept: Accuracy scales with sqrt(N). Doubling data doesn't double accuracy.
# ==========================================
section_header("INFORMATION")

data_sizes = [50, 100, 200, 400, 800, 1600, 3200]
accuracies = []

# Using a simple linear model to observe data scaling effects clearly
X_info, y_info = make_classification(n_samples=5000, n_features=20, n_informative=10, random_state=42)

print(f"{'Data Size':<10} | {'Accuracy':<10}")
print("-" * 25)

for n in data_sizes:
    # Subset the data
    X_sub, _, y_sub, _ = train_test_split(X_info, y_info, train_size=n, stratify=y_info, random_state=42)
    
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_sub, y_sub)
    
    # Evaluate on a large fixed holdout set to estimate true performance
    acc = model.score(X_info[3200:], y_info[3200:])
    accuracies.append(acc)
    print(f"{n:<10} | {acc:.4f}")

print("\n-> Notice how accuracy gains slow down massively as N increases.")
print("-> This mirrors modern 'Scaling Laws' in LLMs.")


# ==========================================
# 3. LIKELIHOOD (Optimization via Loss)
# Concept: We don't find 'truth'; we maximize the likelihood (minimize Negative Log Likelihood).
# ==========================================
section_header("LIKELIHOOD")

# A tiny PyTorch Neural Network
# In modern ML, 'Likelihood' is usually expressed as the Loss Function.
# Maximizing Likelihood == Minimizing Cross Entropy Loss.

# Fake data: 5 samples, 3 classes
inputs = torch.randn(5, 4, requires_grad=True)
targets = torch.tensor([0, 2, 1, 0, 2]) # True class indices

# The Model (Linear layer)
criterion = nn.CrossEntropyLoss() # This IS the Negative Log Likelihood calculation

# Forward pass (Calculate Probabilities)
outputs = inputs @ torch.randn(4, 3) 
loss = criterion(outputs, targets)

print(f"Calculated Loss (Negative Log Likelihood): {loss.item():.4f}")
print("-> Training a Neural Net is simply adjusting weights to minimize this value,")
print("   thereby maximizing the probability of the observed data.")


# ==========================================
# 4. INTERCOMPARISON (Internal Validation)
# Concept: Using the data to check itself (Cross-Validation).
# ==========================================
section_header("INTERCOMPARISON")

X_cv, y_cv = make_classification(n_samples=1000, random_state=42)
model_cv = SGDClassifier(random_state=42)

# 5-Fold Cross Validation
# We split the universe of data into 5 parallel universes to check stability
scores = cross_val_score(model_cv, X_cv, y_cv, cv=5)

print(f"Cross-Validation Scores: {scores}")
print(f"Mean Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
print("-> We determine significance internally, without an external 'Gold Standard'.")


# ==========================================
# 5. REGRESSION (Regularization)
# Concept: 'Regression to the Mean'. Preventing the model from chasing outliers (Overfitting).
# ==========================================
section_header("REGRESSION")

# Generate data with a linear trend + noise
X_reg = np.sort(np.random.rand(20, 1) * 10, axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.5, X_reg.shape[0])

# 1. Complex Model (High Degree Polynomial - Overfitting)
# It tries to touch every single data point (chasing the outliers)
poly = PolynomialFeatures(degree=15)
model_overfit = make_pipeline(poly, LinearRegression())
model_overfit.fit(X_reg, y_reg)

# 2. Regularized Model (Ridge - Regression to the Mean)
# We add a penalty (alpha) to force coefficients to be small/simpler
model_reg = make_pipeline(poly, Ridge(alpha=1.0))
model_reg.fit(X_reg, y_reg)

# Test on new points
X_test_reg = np.linspace(0, 10, 100)[:, np.newaxis]
print("Comparing models on unseen data...")
print(f"Overfit Model Variance (coeff sum): {np.sum(np.abs(model_overfit.steps[1][1].coef_)):.2f}")
print(f"Regularized Model Variance (coeff sum): {np.sum(np.abs(model_reg.steps[1][1].coef_)):.2f}")
print("-> Regularization forces the model to ignore extreme idiosyncrasies of the training set.")


# ==========================================
# 6. DESIGN (Randomization)
# Concept: The design of the data stream matters. Shuffling is crucial for SGD.
# ==========================================
section_header("DESIGN")

# Create a dataset that is SORTED by class (all Class 0, then all Class 1)
X_des, y_des = make_classification(n_samples=1000, random_state=42)
sorted_indices = np.argsort(y_des)
X_sorted, y_sorted = X_des[sorted_indices], y_des[sorted_indices]

# Model 1: Training on sorted data (Bad Design)
# We turn off shuffling to simulate bad experimental design
clf_bad = SGDClassifier(shuffle=False, max_iter=1, warm_start=True, random_state=42)
acc_bad_history = []

# Model 2: Training on shuffled data (Good Design / Randomized Trial)
clf_good = SGDClassifier(shuffle=True, max_iter=1, warm_start=True, random_state=42)
acc_good_history = []

# Simulate training batches
for i in range(0, 1000, 100):
    batch_X, batch_y = X_sorted[i:i+100], y_sorted[i:i+100] # Sequential batch
    clf_bad.partial_fit(batch_X, batch_y, classes=[0, 1])
    
    # For the good model, we pick a random batch from the whole set
    rand_idx = np.random.randint(0, 1000, 100)
    clf_good.partial_fit(X_sorted[rand_idx], y_sorted[rand_idx], classes=[0, 1])

print(f"Final Accuracy (Sorted/Bad Design): {clf_bad.score(X_sorted, y_sorted):.4f}")
print(f"Final Accuracy (Randomized/Good Design): {clf_good.score(X_sorted, y_sorted):.4f}")
print("-> Without randomization (Design), the model forgets the beginning as it learns the end.")


# ==========================================
# 7. RESIDUAL (Boosting)
# Concept: Analyzing what's left over (The Residuals) to find structure.
# ==========================================
section_header("RESIDUAL")

# Boosting explicitly trains on residuals
# Iteration 1: Train on Data -> Get Errors (Residuals)
# Iteration 2: Train on Residuals -> Get New Residuals
# ...

X_res, y_res = make_regression(n_samples=200, noise=10, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_res, y_res)

# A Boosting Regressor does this automatically
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train_r, y_train_r)

print(f"Gradient Boosting R2 Score: {gb.score(X_test_r, y_test_r):.4f}")

# Let's peek under the hood at the "staged" improvements
errors = [mean_squared_error for mean_squared_error in gb.train_score_]
print(f"Initial Error (MSE): {errors[0]:.2f}")
print(f"Final Error (MSE): {errors[-1]:.2f}")
print("-> The model iteratively discovered structure hidden in the residuals of the previous step.")