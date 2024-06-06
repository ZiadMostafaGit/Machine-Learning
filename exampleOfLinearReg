Sure! Let's go through a complete example of linear regression using gradient descent, covering every step in detail. We will use a small dataset for simplicity and perform linear regression to fit a line \( y = mx + c \).

### Example Dataset

| Input (x) | Output (y) |
|-----------|------------|
| 1         | 2          |
| 2         | 3          |
| 3         | 5          |
| 4         | 7          |
| 5         | 11         |

We want to find the best-fitting line \( y = mx + c \).

### Steps for Linear Regression using Gradient Descent

#### Step 1: Initialize Parameters
Let's start with initial guesses for the parameters \( m \) and \( c \):
- \( m = 0 \)
- \( c = 0 \)

We'll also choose a learning rate (\(\alpha\)) of 0.01.

#### Step 2: Compute the Cost Function
The cost function (Mean Squared Error) for our data is:
\[ \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i})^2 \]
where \( \hat{y_i} = mx_i + c \) is the predicted value.

#### Step 3: Gradient Descent Algorithm
We'll use Batch Gradient Descent in this example.

##### Gradient Calculation
The gradients of the cost function with respect to \( m \) and \( c \) are:
\[ \frac{\partial \text{Cost}}{\partial m} = -\frac{2}{N} \sum_{i=1}^N x_i (y_i - (mx_i + c)) \]
\[ \frac{\partial \text{Cost}}{\partial c} = -\frac{2}{N} \sum_{i=1}^N (y_i - (mx_i + c)) \]

##### Update Parameters
Update \( m \) and \( c \) using the gradients:
\[ m_{\text{new}} = m - \alpha \frac{\partial \text{Cost}}{\partial m} \]
\[ c_{\text{new}} = c - \alpha \frac{\partial \text{Cost}}{\partial c} \]

##### Repeat
We'll repeat these steps for a certain number of iterations or until the cost function converges.

### Iteration Example
Let's go through the first iteration in detail.

**Initial Values**:
- \( m = 0 \)
- \( c = 0 \)
- \( \alpha = 0.01 \)

#### Calculate Gradients
1. Compute the predictions with the current parameters:
   \[
   \hat{y} = [0 \cdot 1 + 0, 0 \cdot 2 + 0, 0 \cdot 3 + 0, 0 \cdot 4 + 0, 0 \cdot 5 + 0] = [0, 0, 0, 0, 0]
   \]

2. Compute the gradients:
   \[
   \frac{\partial \text{Cost}}{\partial m} = -\frac{2}{5} [(1(2-0) + 2(3-0) + 3(5-0) + 4(7-0) + 5(11-0))]
   \]
   \[
   \frac{\partial \text{Cost}}{\partial c} = -\frac{2}{5} [(2-0) + (3-0) + (5-0) + (7-0) + (11-0)]
   \]

   Calculate these sums:
   \[
   \frac{\partial \text{Cost}}{\partial m} = -\frac{2}{5} [2 + 6 + 15 + 28 + 55] = -\frac{2}{5} \cdot 106 = -42.4
   \]
   \[
   \frac{\partial \text{Cost}}{\partial c} = -\frac{2}{5} [2 + 3 + 5 + 7 + 11] = -\frac{2}{5} \cdot 28 = -11.2
   \]

#### Update Parameters
3. Update \( m \) and \( c \):
   \[
   m_{\text{new}} = m - \alpha \cdot (-42.4) = 0 + 0.01 \cdot 42.4 = 0.424
   \]
   \[
   c_{\text{new}} = c - \alpha \cdot (-11.2) = 0 + 0.01 \cdot 11.2 = 0.112
   \]

#### Repeat for Subsequent Iterations
We repeat these steps for more iterations. Here, we'll demonstrate a few more iterations to show the progress.

### Second Iteration
**Updated Values**:
- \( m = 0.424 \)
- \( c = 0.112 \)

#### Calculate Gradients
1. Compute the predictions with the updated parameters:
   \[
   \hat{y} = [0.424 \cdot 1 + 0.112, 0.424 \cdot 2 + 0.112, 0.424 \cdot 3 + 0.112, 0.424 \cdot 4 + 0.112, 0.424 \cdot 5 + 0.112] = [0.536, 0.96, 1.384, 1.808, 2.232]
   \]

2. Compute the gradients:
   \[
   \frac{\partial \text{Cost}}{\partial m} = -\frac{2}{5} [(1(2-0.536) + 2(3-0.96) + 3(5-1.384) + 4(7-1.808) + 5(11-2.232))]
   \]
   \[
   \frac{\partial \text{Cost}}{\partial c} = -\frac{2}{5} [(2-0.536) + (3-0.96) + (5-1.384) + (7-1.808) + (11-2.232)]
   \]

   Calculate these sums:
   \[
   \frac{\partial \text{Cost}}{\partial m} = -\frac{2}{5} [1.464 + 4.08 + 10.848 + 20.768 + 43.84] = -\frac{2}{5} \cdot 81 = -32.4
   \]
   \[
   \frac{\partial \text{Cost}}{\partial c} = -\frac{2}{5} [1.464 + 2.04 + 3.616 + 5.192 + 8.768] = -\frac{2}{5} \cdot 21.08 = -8.432
   \]

#### Update Parameters
3. Update \( m \) and \( c \):
   \[
   m_{\text{new}} = 0.424 - 0.01 \cdot (-32.4) = 0.424 + 0.324 = 0.748
   \]
   \[
   c_{\text{new}} = 0.112 - 0.01 \cdot (-8.432) = 0.112 + 0.084 = 0.196
   \]

### Third Iteration
**Updated Values**:
- \( m = 0.748 \)
- \( c = 0.196 \)

#### Calculate Gradients
1. Compute the predictions with the updated parameters:
   \[
   \hat{y} = [0.748 \cdot 1 + 0.196, 0.748 \cdot 2 + 0.196, 0.748 \cdot 3 + 0.196, 0.748 \cdot 4 + 0.196, 0.748 \cdot 5 + 0.196] = [0.944, 1.692, 2.44, 3.188, 3.936]
   \]

2. Compute the gradients:
   \[
   \frac{\partial \text{Cost}}{\partial m} = -\frac{2}{5} [(1(2-0.944) + 2(3-1.692) + 3(5-2.44) + 4(7-3.188) + 5(11-3.936))]
   \]
   \[
   \frac{\partial \text{Cost}}{\partial c} = -\frac{2}{5} [(2-0.944) + (3-1.692) + (5-2.44) + (7-3.188) + (11-3.936)]
   \]

   Calculate these sums:
   \[
   \frac{\partial \text{Cost}}{\partial m} = -\frac{2}{5} [1.056 + 2.616 + 7.68 + 15.248 + 35.32] = -\frac{2}{5} \cdot 61.92 = -24.768
   \]
   \[
   \frac{\partial \text{Cost}}{\partial c} = -\frac{2}{5} [

1.056 + 1.308 + 2.56 + 3.812 + 7.064] = -\frac{2}{5} \cdot 15.8 = -6.32
   \]

#### Update Parameters
3. Update \( m \) and \( c \):
   \[
   m_{\text{new}} = 0.748 - 0.01 \cdot (-24.768) = 0.748 + 0.24768 = 0.99568
   \]
   \[
   c_{\text{new}} = 0.196 - 0.01 \cdot (-6.32) = 0.196 + 0.0632 = 0.2592
   \]

### Summary of Updates
- Initial: \( m = 0 \), \( c = 0 \)
- After 1st iteration: \( m = 0.424 \), \( c = 0.112 \)
- After 2nd iteration: \( m = 0.748 \), \( c = 0.196 \)
- After 3rd iteration: \( m = 0.99568 \), \( c = 0.2592 \)

### Final Thoughts
The parameters \( m \) and \( c \) are updated iteratively to minimize the cost function. The process continues until the cost function converges to a minimum value.

This complete example demonstrates the steps involved in performing linear regression using gradient descent. By following these steps with more iterations, the model will converge to the optimal values of \( m \) and \( c \) that best fit the given data.
