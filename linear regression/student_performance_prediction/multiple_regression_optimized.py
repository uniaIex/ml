import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Load and prepare data
dataset = pd.read_csv("student_performance.csv")
features = torch.tensor(
    dataset[['hours_studied', 'prev_score', 'sleep_h', 'practiced_exams']].values,
    dtype=torch.float32
)
target = torch.tensor(dataset['performance_index'].values, dtype=torch.float32).view(-1, 1)

# Define model and optimizer
model = nn.Linear(4, 1)  # 4 input features, 1 output
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001)

# Training loop
loss_history = []
for _ in tqdm(range(5000)):
    optimizer.zero_grad()
    predictions = model(features)
    loss = criterion(predictions, target)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

# Extract learned parameters
a, b, c, d = model.weight[0].detach().numpy()
e = model.bias[0].detach().numpy()

plt.figure(figsize=(5,5))
plt.scatter(target, predictions.detach(), c='k', s=4)
plt.xlabel('performance index'), plt.ylabel('predicted');
x = y = range(300)
plt.plot(x,y, c='brown')
plt.xlim(0,120), plt.ylim(0,120);
plt.text(80,60, f'e     = {e.item():2.4f}', fontsize=10);
plt.text(80,50, f'hours studied     = {a.item():2.4f}', fontsize=10);
plt.text(80,40, f'previous exam scores   = {b.item():2.4f}', fontsize=10);
plt.text(80,30, f'hours slept = {c.item():2.4f}', fontsize=10);
plt.text(80,20, f'practiced exams  = {d.item():2.4f}', fontsize=10);
plt.show()

print(f'MSE: {loss_history[-1]:.4f}')
print(f'hours studied: {a:.4f}')
print(f'previous exam scores: {b:.4f}')
print(f'hours slept: {c:.4f}')
print(f'practiced exams: {d:.4f}')
print(f'bias: {e:.4f}')

with torch.no_grad():
    predictions = model(features).numpy()

# Convert tensors to numpy arrays
target_np = target.numpy().flatten()
features_np = features.numpy()
residuals = target_np - predictions.flatten()

# 1. Feature-Target Relationships
feature_names = ['Hours Studied', 'Previous Score', 'Sleep Hours', 'Practiced Exams']
plt.figure(figsize=(15, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    sns.regplot(x=features_np[:, i], y=target_np, scatter_kws={'alpha':0.3})
    plt.xlabel(feature_names[i])
    plt.ylabel('Performance Index')
plt.tight_layout()
plt.show()

# 2. Residual Analysis
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')

plt.subplot(1, 2, 2)
sns.scatterplot(x=predictions.flatten(), y=residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.tight_layout()
plt.show()

# 3. Coefficient Importance
coefficients = model.weight[0].detach().numpy()
intercept = model.bias[0].detach().numpy()

plt.figure(figsize=(10, 4))
bars = plt.bar(feature_names + ['Intercept'], list(coefficients) + [intercept])
plt.bar_label(bars, fmt='%.2f')
plt.title('Feature Weights in Trained Model')
plt.ylabel('Coefficient Value')
plt.axhline(0, color='black', linewidth=0.8)
plt.show()

# 4. Enhanced Actual vs Predicted Plot
plt.figure(figsize=(6, 6))
sns.scatterplot(x=target_np, y=predictions.flatten(), alpha=0.4)
plt.plot([0, 120], [0, 120], '--', color='orange')
plt.xlabel('Actual Performance Index')
plt.ylabel('Predicted Performance Index')
plt.title('Model Predictions vs Ground Truth')
plt.text(10, 100, f'R² Score: {r2_score(target_np, predictions):.2f}', 
         bbox=dict(facecolor='white', alpha=0.9))
plt.grid(alpha=0.2)
plt.xlim(0, 120)
plt.ylim(0, 120)
plt.show()