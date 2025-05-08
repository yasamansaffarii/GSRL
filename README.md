## Overview
This repository contains the implementation of the framework proposed in the paper:

**"GSRL: A Graph-based State Representation learning in Episodic RL for Task-Oriented Dialogue Systems"**  
Authors: Yasaman Saffari, Javad Salimi Sartakhti  
Affiliation: Department of Electrical and Computer Engineering, University of Kashan, Kashan, Iran  
Contact: [y.saffari@grad.kashanu.ac.ir](mailto:saffari@kashanu.ac.ir), [salimi@kashanu.ac.ir](mailto:salimi@kashanu.ac.ir)

---

--- 
## Requirements

**Motivation**: Traditional reinforcement learning (RL) methods often overlook the topological structure of state transitions, which can carry rich information critical for learning better policies, especially in episodic environments like dialogue systems.

**Problem**: Most graph representation learning (GRL) methods are transductive and fail to generalize to unseen states, limiting their use in dynamic RL settings.

**Novelty 1**: Introduced topological state representations by modeling state transitions as a graph, capturing structural patterns beyond raw features.

**Novelty 2**: Proposed an inductive GRL framework by applying clustering on the transition graph before learning embeddings, enabling generalization to unseen or dynamic states.

Novelty 3: Designed a task-aware fine-tuning mechanism using TD error to dynamically update state embeddings based on task-specific feedback.
---

## 🔹 Framework Features 

- **State Transition Graph Construction**  
  - Constructed a directed graph where **nodes represent states** and **edges capture observed transitions** in the environment.

- **Graph Clustering**  
  - Applied **unsupervised clustering** (K-means) to group **topologically similar states**, reducing graph size and sparsity.

- **Inductive Graph Representation Learning (GRL)**  
  - Employed **GraphSAGE and Node2Vec** to learn **node embeddings**.

- **Dimensionality Reduction before GRL**  
  - Performed **clustering before GRL** to reduce the number of nodes, allowing more efficient and stable embedding learning over **cluster-level graphs** and enabling generalization of GRL to **unseen or dynamically emerging states**.

- **Architecture Overview**  
  - **Input**: Raw state transitions  
  - → Graph construction  
  - → Graph clustering  
  - → GRL (e.g., GraphSAGE)  
  - → Topological embeddings  
  - → Used in **Q-learning** or other RL algorithms as enriched state representations

- **TD Error–Based Fine-Tuning**  
  - Introduced a **task-aware dynamic update** mechanism using **temporal difference (TD) error** to fine-tune embeddings during training.

- **Support for Episodic Environments**  
  - Designed for **episodic, sparse-reward RL tasks**, such as **goal-oriented dialogue systems**, where topological structure provides critical guidance.

---
## Repository Structure
- `deep_dialog/`: Contains the source code for the framework.
- `data/`: Includes the preprocessed movie ticket booking, restaurant reservation, and taxi ordering dataset.
- `script/`: Scripts for running experiments and generating results.
- `README.md`: This file.
- `finalgsrl.ipynb`: Run codes for train GRL and clustering algorithms.
---
## Requirements
To reproduce the experiments, install the following dependencies:
- Python >= 3.8
- PyTorch >= 1.10
- Transformers >= 4.0
- NumPy, Scikit-learn, and Matplotlib
---
## Train the Agent:
   - **Movie Ticket Booking domain:**  
     ```bash
     bash script/movie.sh
     ```
   - **Restaurant Reservation domain:**  
     ```bash
     bash script/restaurant.sh
     ```
   - **Taxi Ordering domain:**  
     ```bash
     bash script/taxi.sh
     ```
---
## Hyperparameter Settings
to change N(dimension of GRL)  =>  go to agent_dqn.py, line 60
to change grl algorithm or k  =>  go to agent_dqn.py line 205, Topol_rep.

--- 
## Citation
If you use this work, please cite:

Salimi Sartakhti, Javad and Saffari, Yasaman, Gsrl: A Graph-Based State Representation Learning In Episodic Rl for Task-Oriented Dialogue Systems. Available at SSRN: https://ssrn.com/abstract=5139155 or http://dx.doi.org/10.2139/ssrn.5139155

---
