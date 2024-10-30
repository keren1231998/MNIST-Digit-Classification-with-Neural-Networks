# MNIST-Digit-Classification-with-Neural-Networks

## Authors
- **Keren Gorelik** - 206749731
- **Lior Kobi** - 318663465

## Project Overview
This project is part of a deep learning assignment focused on implementing and evaluating a neural network using PyTorch on the MNIST dataset for digit classification. We conducted three experiments to explore the impact of different network configurations, including:

1. A baseline model without normalization.
2. A model with Batch Normalization.
3. A model with L2 Regularization.

## Network Structure
- **Model Architecture:** Input layer with four hidden layers structured as `[784, 20, 7, 5, 10]`.
- **Training Parameters:**
  - Batch size: 64
  - Learning rate: 0.009
  - Total training iterations: 100

## Experiments

1. **Experiment 1 - Baseline Model (No Normalization):**
   - **Configuration:** No batch normalization or regularization.
   - **Results:** High accuracy (92.80% on the test set), but required a longer training time.

2. **Experiment 2 - Batch Normalization Enabled:**
   - **Configuration:** Batch normalization added after activation functions.
   - **Results:** Faster training time with 86.55% test accuracy. Some accuracy loss, suggesting a need for fine-tuning.

3. **Experiment 3 - L2 Regularization Enabled (λ = 0.01):**
   - **Configuration:** Added L2 regularization to penalize large weights.
   - **Results:** High accuracy (93.57% on test set) with reduced training time, balancing generalization and performance effectively.

## Results Summary
Each configuration provided insights into how regularization and normalization impact model performance:
- **Without Batch Normalization:** High accuracy but longer training time.
- **With Batch Normalization:** Improved training speed, but slightly lower accuracy.
- **With L2 Regularization:** Best balance of high accuracy, training speed, and generalization.

## Dependencies
- Python 3.x
- PyTorch
- NumPy

## How to Run
1. Clone the repository.
2. Install the required packages:
   ```bash
   pip install torch numpy
   ```
3. Run the training script:
   ```bash
   python main.py
   ```
4. View the training results and accuracy metrics upon completion.

