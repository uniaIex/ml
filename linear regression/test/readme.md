# Multiple Linear Regression with PyTorch

This project demonstrates a manual implementation of **multiple linear regression** using PyTorch to predict student performance based on study habits and other factors. The goal is to illustrate the fundamentals of gradient descent and tensor operations in PyTorch, while emphasizing transparency in model training.

## Dataset
The dataset `student_performance.csv` contains the following features:
- `hours_studied`: Number of hours studied.
- `prev_score`: Previous exam score (baseline performance).
- `sleep_h`: Average hours of sleep per night.
- `practiced_exams`: Number of practice exams completed.
- `performance_index`: Target variable representing student performance (higher values indicate better performance).

The model uses all features except `extra_activ` (excluded for simplicity) to predict `performance_index`.

## Model Structure
The linear regression model is defined as:
\[
\text{performance\_index} = a \cdot \text{hours\_studied} + b \cdot \text{prev\_score} + c \cdot \text{sleep\_h} + d \cdot \text{practiced\_exams} + e
\]
where \(a, b, c, d\) are coefficients, and \(e\) is the intercept. Coefficients are initialized randomly and optimized via gradient descent.

## Key Steps
1. **Data Preparation**:  
   - Features and target are converted to PyTorch tensors.
   - A correlation matrix is generated to analyze variable relationships.

2. **Training**:
   - **Loss Function**: Mean Squared Error (MSE).
   - **Optimization**: Manual gradient descent over 5,000 epochs with a learning rate of \(1 \times 10^{-6}\).
   - Gradients are computed using PyTorch's autograd system.

3. **Visualization**:
   - **Prediction vs. Actual Plot**: Compares model predictions against true values before and after training.
   - **Loss Curve**: Tracks MSE reduction over epochs.
   - Coefficient values are displayed dynamically on the prediction plot.

## Results
- The model starts with random coefficients (high initial loss ~2143).
- After training, the loss decreases significantly, and predictions align closer to the actual performance values (ideal line \(y = x\)).
- Final coefficients reflect the learned relationships between features and the target.

## Dependencies
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `torch`, `tqdm`

## Usage
1. Clone the repository and install dependencies.
2. Ensure `student_performance.csv` is in the working directory.
3. Run `multiple_regression.py`:
   ```bash
   python multiple_regression.py


# Updated Multiple Linear Regression with PyTorch Optim Module

This project demonstrates **multiple linear regression** using PyTorch's built-in optimization tools to predict student performance. The code now leverages PyTorch's `nn.Module`, `nn.Linear`, and `torch.optim` for streamlined training.

## Key Improvements
- Uses PyTorch's `nn.Linear` for parameter management.
- Employs `torch.optim.SGD` for gradient descent.
- Fixes the intercept gradient bug from the original code.
- Simplifies training loops and gradient updates.

''' Key Changes Explained
Data Handling
Features are combined into a single tensor of shape (n_samples, 4) for compatibility with nn.Linear.

Model Definition

nn.Linear(4, 1) replaces manual coefficient management.

Automatically tracks weights (for features) and bias (intercept e).

Optimization

Uses torch.optim.SGD to handle parameter updates.

Gradient zeroing and updates are automated via optimizer.zero_grad() and optimizer.step().

Loss Calculation
Built-in nn.MSELoss() replaces the manual MSE implementation.

'''
---