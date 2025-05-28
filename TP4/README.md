# TP4: Self-Organizing Map, Oja’s Rule & Hopfield Network
**Authors:** Group 7 
- Mona Helness
- Lilian Michalak
- Camilla Adriazola Johannessen
- Lukas Wittich

This repository contains implementations for **Exercise 1** (Kohonen SOM & Oja’s Rule) and **Exercise 2** (Hopfield Network for 5×5 patterns) of the SIA TP4 assignment

## Directory Structure

```
/                        # Project root
├── README.md            # This overview
├── exercise_1/
│   ├── data/europe.csv             # Europe dataset for Exercise 1
│   └── src/
│       ├── utils.py               # Data loading & custom standardization
│       ├── kohonen.py             # SelfOrganizingMap implementation
│       ├── oja.py              # OjaNetwork implementation
│       └── exercise_1.ipynb     # Notebook containing plots and visualisations for exercise 1
└── exercise_2/src/
    ├── patterns.py            # flatten/unflatten & noise functions
    ├── exercise_2.ipynb       # Hopfield workflow, static & animated views
    └── hopfield.py            # HopfieldNetwork + plotting & animation

```


## Usage

1. Navigate to the directory:
    ```bash
    cd <directory>
    ```

2. Read the Jupyter Notebooks:
    - ['exercise_1.ipynb'](exercise_1/src/exercise_1.ipynb)
    Contains Plots for Exercise 1 of the TP
    - ['exercise_2.ipynb'](exercise_2/src/exercise_2.ipynb)
    Contains Plots for Exercise 2 of the TP
