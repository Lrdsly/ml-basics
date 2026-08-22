# Softmax Regression

## What is Softmax Regression?
While logistic regression is used for binary classification (two classes), softmax regression (also known as multinomial logistic regression) is a generalization designed for multi-class classification. It determines the probability that a given sample belongs to one of multiple discrete categories.

**Example: Predict whether an animal in the photo is a dog, a cat, or a bird.**

---

## Model Architecture

**Parameter Definitions:**
- $N$: The total number of samples in the dataset.
- $i$: The index of each sample (row) in the dataset.
- $X$: The input matrix where each row represents the features of a sample.
- $W$: The weight matrix, where columns correspond to different classes and rows correspond to features.
- $b$: The bias vector, providing a separate intercept term for each class.
- $z$: The raw linear output matrix ($z = XW + b$), representing unnormalized scores (logits) for each class.
- $\hat{y}$: The predicted probabilities for each class after applying the softmax function.
- $y$: The actual one-hot encoded true labels.
- $\alpha$: The learning rate.

---

## How Softmax Works
For each sample, the model outputs a vector of raw scores. To convert these scores into valid probabilities (where every value is between $0$ and $1$ and the sum across all classes equals $1$), the **Softmax function** is applied:

$$\hat{y}_j = \frac{e^{z_j}}{\sum_{k} e^{z_k}}$$

**Why exponential?**
1. It maps any real-valued score to a strictly positive range ($> 0$), avoiding negative probabilities.
2. It amplifies differences, making the model more confident and decisive about the highest-scoring class.

---

## Optimization & Gradient Descent

**Loss Function (Cross-Entropy Loss):**
$$J(W, b) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j} y_{ij}\log(\hat{y}_{ij})$$

**Gradients:**
Just like in logistic regression, we compute the gradient of the loss with respect to our parameters using the error matrix ($\hat{y} - y$):

- **Weight Gradient:**
  $$\nabla_W J = \frac{1}{N} X^T (\hat{y} - y)$$

- **Bias Gradient:**
  $$\nabla_b J = \frac{1}{N} \sum (\hat{y} - y)$$

**Gradient Descent Updates:**
$$W = W - \alpha \cdot \nabla_W J$$
$$b = b - \alpha \cdot \nabla_b J$$