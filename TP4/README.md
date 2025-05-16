# Practical Assignment 4 – Unsupervised Learning  
**Artificial Intelligence Systems 2025**

## Overview

This project contains solutions for Practical Assignment 4 of the course *Sistemas de Inteligencia Artificial*. It covers two unsupervised learning tasks:

1. Clustering socio-economic data of European countries (Exercise 1).
2. Pattern recognition using Hopfield networks (Exercise 2).

---

## 1. Europe Dataset Analysis (`exercise1/`)

### 📁 Structure
- `data/europe.csv`: Dataset with 28 countries and socio-economic features.
- `src/ex1.ipynb`: Kohonen network (SOM) implementation and visualization.
- `src/ex2.ipynb`: Oja’s rule implementation and PCA comparison.

### 1.1 Kohonen Network

- Trained a **Self-Organizing Map (SOM)** to group countries with similar geopolitical, economic, and social traits.
- Visualized:
  - Country distribution over the SOM grid.
  - Average distances between neighboring neurons.
  - Number of countries assigned to each neuron.

### 1.2 Oja’s Rule Neural Network

- Implemented a single-layer neural network trained with **Oja's learning rule**.
- Computed the **first principal component** of the dataset.
- Compared the result with PCA from `scikit-learn` and interpreted the weight vector.

---

## 2. Pattern Recognition with Hopfield Network (`exercise2/`)

### 📁 Files
- `exercise2/ex1.ipynb`: Implementation of Hopfield network with binary letter patterns.
- `exercise2/hopfield_steps.gif`: GIF visualization of the step-by-step recall process.

### 2.1 Hopfield Network

- Encoded 4 letters (`A`, `H`, `J`, `T`) as **5×5 binary matrices** using values −1 and 1.
- Trained a Hopfield network to:
  - Recover noisy input patterns through asynchronous updates.
  - Visualize the convergence step-by-step (via GIF).
  - Detect a **spurious state** when input noise is too high.
  - Evaluate reconstruction quality using similarity metrics.

---

## How to Run

1. Clone the repository.
2. Open the Jupyter notebooks under `exercise1/src/` and `exercise2/` folders.
3. Make sure required libraries are installed:
   - `numpy`, `matplotlib`, `scikit-learn`, `imageio`, `PIL` (pillow)

---

## Deliverables

- 📁 Repository with organized code and outputs
- 🎞️ Visualization GIFs and plots
- 📄 `README.md` (this file)
- 📝 Configuration and environment info
- 🧾 Git commit hash
- 🎤 Presentation (submitted separately via Campus)

---

