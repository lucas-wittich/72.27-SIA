# TP3: Perceptron

**Course:** SIA - Artificial Intelligence Systems (72.27)  
**Authors:** Group 7 
- Mona Helness
- Lilian Michalak
- Camilla Adriazola Johannessen
- Lukas Wittich


## Table of Contents

1. [Project Overview](#project-overview)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Directory Structure](#directory-structure)  
5. [Usage](#usage)  
   - [Exercise 1: Simple Perceptron](#exercise-1-simple-perceptron)  
   - [Exercise 2: Linear vs Nonlinear Perceptron](#exercise-2-linear-vs-nonlinear-perceptron)  
   - [Exercise 3: Multilayer Perceptron](#exercise-3-multilayer-perceptron)  
6. [Results & Evaluation](#results--evaluation)  
7. [Hyperparameters & Architecture Choices](#hyperparameters--architecture-choices)  
8. [Conclusions](#conclusions)  
9. [References](#references)  


## Project Overview

- **Exercise 1:** Implement a simple perceptron (step activation) to solve AND and XOR.  
- **Exercise 2:** Compare linear vs. nonlinear perceptron on the provided CSV dataset; evaluate generalization via cross-validation.  
- **Exercise 3:** Build a multilayer perceptron with GD, Momentum, and Adam optimizers; tasks include XOR validation, parity detection, and digit recognition (with noise testing).


## Prerequisites

- Python 3.8+  
- Packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `jupyter`       (if you wish to run notebooks)



## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/lucas-wittich/72.27-SIA
   cd TP3
   ```

2. Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```
---

## Directory Structure
```bash
TP3/
├── README.md
├── requirements.txt
│
├── exercise1/
│   └── src/
│       ├── perceptron.py
│       └── test_ex1.ipynb
│
├── exercise2/
│   ├── data/
│   │   └── TP3-ej2-conjunto.csv
│   └── src/
│       ├── cross_validation.py
│       ├── perceptron.py
│       └── test_ex1.ipynb
│
└── exercise3_multilayer_perceptron/
    ├── data/
    │   ├── noisy_digits/
    │   └── TP3-ej3-digitos.txt
    └── src/
        ├── optimizer.py
        ├── perceptron.py
        ├── test_ex3.ipynb
        ├── test_XOR.py
        ├── train_digits.py
        └── train_parity.py
```


## Usage

### Exercise 1

1. Navigate to the directory:
    ```bash
    cd exercise_1/src
    ```

2. Run the code:
    - ['perceptron.py'](exercise_1/src/perceptron.py)
    For a simple evaluation of the perceptron which precents weight evolution and predictions for AND and XOR.
    ```bash
    python perceptron.py
    ```
    - ['text_ex1.ipynb'](exercise_1/src/test_ex1.ipynb)
    Evaluates AND and XOR with various plots.

### Exercise 2

1. Navigate to the directory:
    ```bash
    cd exercise_2/src
    ```

2. Run the code:
    - ['perceptron.py'](exercise_2/src/perceptron.py)
    For a simple evaluation of the perceptron with presents the accuracy of the Linear and Nonlinear perceptrons, with sigmoid and tanh activation functions in the Nonlinear one.
    ```bash
    python perceptron.py
    ```
    - ['cross_validation.py'](exercise_2/src/cross_validation.py)
    It computes the cross validation over 5 folds for the Linear Perceptron.
    ```bash
    python cross_validation.py
    ```
    - ['test_ex2.ipynb'](exercise_2/src/test_ex2.ipynb)
    A Jupyter notebook containing plots and comparisons of the perceptrons.


### Exercise 3

1. Navigate to the directory:
    ```bash
    cd exercise_2/src
    ```

2. Run the code:
    ```bash
    python <file_to_run.py>
    ```
    - ['test_XOR.py'](exercise_3/src/test_XOR.py)
    Script that tests the MLP on XOR and returns the output in the console
    - ['test_digits.py'](exercise_3/src/test_digits.py)
    Script that tests the digit recognition for simple and noise affected pixels.
    - ['test_parity.py'](exercise_3/src/test_parity.py)
    Script that tests the MLP in terms of parity detection.
    - As well as that there are three notebooks containing the plots for the sub exercises in exercise 3.



## Results and Evaluation

### Exercise 1
#### What can you say about the types of problems the simple step perceptron can solve, in relation to the logical problems given in the task?
The perceptron had no problems learning the AND function however it could not solve the XOR, this being due to the fact that XOR is not a linearly separable operation. The AND function is linearly separable as you can draw a straight line that separates the positive output (1) from the negatives (-1), which is not the case with XOR. To solve such a problem, a multi layer perceptron would be required as it introduces non-linearity to the decision boundaries it can represent.

--- 

### Exercise 2
#### How would you choose the best training set?
To select the optimal training set, we employ stratified k-fold cross-validation. First, we split the full dataset into k equally-sized folds, ensuring that each fold preserves the overall class balance. Then, for each of the k iterations, we use k – 1 folds for training and hold out the remaining fold for validation. We repeat this process across all folds and compute the average validation accuracy (or loss) for each candidate configuration (e.g. different thresholds, sampling strategies, feature‐scaling parameters). The training set that yields the highest mean validation performance, while exhibiting low variance across folds, is chosen as the “best” training partition. This approach guarantees that every example in the dataset contributes exactly once to validation, and that our selection is based on a robust, unbiased estimate of out-of-sample performance.

#### What effect does this choice have on generalization?
By using stratified k-fold cross-validation to choose the training set, we achieve three key benefits for generalization:

- **Reduced Overfitting:** Because the model is evaluated repeatedly on unseen folds, it cannot simply memorize a single train/test split’s peculiarities.

- **Bias–Variance Trade-off Control:** The mean validation score reflects the model’s average performance (bias), while the standard deviation across folds measures its stability (variance). A low-variance choice indicates the model generalizes consistently across different subsets.

- **Reliable Performance Estimates:** Averaging over multiple folds smooths out random fluctuations due to any one particular split, giving a more trustworthy estimate of how the model will perform on truly unseen data.
