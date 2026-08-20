# Logistic Regression

## What is Regression ?
Regression is a way to predict a continuous value. The goal is to find a relation among previous datas and guess a numbrical value based on that relation.

**Example: Predict a house price based on it's area**

## What is Classification
In opposite of regression, classification is used to predict a label (Discrete Class/Category). The goal is to guess the category which new data blongs to.

**Example: Predict the animal in the photo is either a cat or a doga.**


### What does Regression Allgorithms do?
This Allgorithms try to pass a line/plane or a curve through the data which decrease the prediction error to minimum value.

**Popular Examples:** 
- Linear Regression: find the best stright line to predict the number
- Polynomial Regression: Use curves for more complex relations
- etc

### What does Logistic Regresion?
Despite it's name, log-reg is a classification allgorithm! It pass the linear output of the model to a non-linear function (Usually Sigmoid) to keep output between 0 and 1. At last it will allocate models across categories


### Model

**Parameter Definitions:**
- $N$: The total number of samples in the dataset.
- $i$: The index of each sample (row) in the dataset.
- $y^{(i)}$: The actual binary value (0 or 1) for the $i$-th sample.
- $\hat{y}^{(i)}$: The model's predicted value for the $i$-th sample, obtained after applying the sigmoid function.
- $J(w,b)$: The overall loss of the model with respect to the weights ($w$) and bias ($b$).
- $w$: The model's weight parameters.
- $b$: The model's bias parameter.
- $x^{(i)}$: The $i$-th row of the input matrix $X$, representing the features of the $i$-th sample.


**Loss Function**

$$J(w,b) = -\frac{1}{N}\sum_{i=1}^{N}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]$$

**Chain Rule**

$$\frac{\partial J}{\partial w} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

where

$$
z = Xw + b
$$

and

$$
\hat{y} = sigmoid(z)
$$

For each sample $i$, where $x^{(i)}$ is the $i$-th row of $X$:

$$\frac{\partial J}{\partial w} = \frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}^{(i)}-y^{(i)}\right)x^{(i)}$$

**Gradient Descent**

$$w = w - \alpha \cdot \nabla_w J$$

where:

- $\alpha$: The learning rate.
- $\nabla_w J$: The gradient of the loss function with respect to the weights $w$.
