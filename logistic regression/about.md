# Learning Objectives

- Identify use cases for performing logistic regression.
- Explain how logistic regression models use the sigmoid function to calculate probability.
- Compare linear regression and logistic regression.
- Explain why logistic regression uses log loss instead of squared loss.
- Explain the importance of regularization when training logistic regression models.

## Sigmoid Functions

As it happens, there is a family of functions called logistic functions whose output has those same characteristics. The standard logistic function, also known as the sigmoid function, has the formula:

\[ f(x) = \frac{1}{1 + e^{-x}} \]

(The curve approaches 0 as \( x \) values decrease to negative infinity, and 1 as \( x \) values increase toward infinity.)

## Transforming Linear Output Using the Sigmoid Function

The following equation represents the linear component of a logistic regression model:

\[ z = b + w_1x_1 + w_2x_2 + \dots + w_nx_n \]

where:
- \( z \) is the output of the linear equation
- \( b \) is the bias
- \( w \) values are the model's learned weights
- \( x \) values are the feature values for a particular example

To obtain the logistic regression prediction, the \( z \) value is then passed to the sigmoid function, yielding a value (a probability) between 0 and 1:

\[ y' = \frac{1}{1 + e^{-z}} \]

where:
- \( y' \) is the output of the logistic regression model
- \( z \) is the linear output

## Training Logistic Regression Models

Logistic regression models are trained using the same process as linear regression models, with two key distinctions:
1. Logistic regression models use **Log Loss** as the loss function instead of squared loss.
2. Applying **regularization** is critical to prevent overfitting.

### Log Loss

In the Linear regression module, you used squared loss (also called L2 loss) as the loss function. Squared loss works well for a linear model where the rate of change of the output values is constant. For example, given the linear model, each time you increment the input value by 1, the output value increases by 3.

However, the rate of change of a logistic regression model is not constant. As you saw in "Calculating a Probability," the sigmoid curve is S-shaped rather than linear. When the log-odds value is closer to 0, small increases in \( z \) result in much larger changes to \( y' \) than when \( z \) is a large positive or negative number.

If you used squared loss to calculate errors for the sigmoid function, as the output got closer and closer to 0 and 1, you would need more memory to preserve the precision needed to track these values.

Instead, the loss function for logistic regression is **Log Loss**. The Log Loss equation returns the logarithm of the magnitude of the change, rather than just the distance from data to prediction.

### Regularization in Logistic Regression

Regularization, a mechanism for penalizing model complexity during training, is extremely important in logistic regression modeling. Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in cases where the model has a large number of features. Consequently, most logistic regression models use one of the following two strategies to decrease model complexity:

1. **L2 Regularization**
2. **Early Stopping**: Limiting the number of training steps to halt training while loss is still decreasing.