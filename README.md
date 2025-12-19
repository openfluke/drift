# DRIFT: Decentralized Real-time Integration & Functional Transfer

> **Status:** Experimental / Active Research
> **Paradigm:** Bio-Mimetic AI / Online Learning
> **Learning Rule:** Non-Backpropagation (Local/Hebbian)

## 🧠 The Philosophy

**DRIFT** is an experimental framework exploring the boundaries of **Decentralized Functional Integration**.

Current AI architectures rely heavily on monolithic networks and global error signals (backpropagation). While powerful, they fail to mimic the modular, emergent nature of biological intelligence. In the brain, specialized regions (e.g., visual cortex vs. prefrontal cortex) operate independently but bind their information together dynamically to solve complex tasks—a phenomenon known as **Functional Integration**.

**The core hypothesis of DRIFT is:**
*Two specialized, heterogeneous neural agents (e.g., a Predictor and a Classifier) can autonomously develop a communication protocol in real-time to solve a composite task that neither can solve alone, without the need for a central "gating" network or global gradient descent.*

## 🎯 Project Goals

The primary goal is to demonstrate **Synergistic Emergence**: where the combined capability of communicating agents is qualitatively higher than the sum of their individual parts.

We are conducting a series of tests to validate:
1.  **Emergent Communication:** Can two agents invent a signaling method to share latent states?
2.  **Task Binding:** Can a "Predictor" and a "Classifier" negotiate a consensus on reality to solve a third, unseen task?
3.  **Online Plasticity:** Can this negotiation happen in *real-time* (online learning) rather than effectively freezing weights after training?

## ⚙️ Architecture & Methodology

DRIFT differs from standard Mixture of Experts (MoE) or GANs in three critical ways:

### 1. Decentralized Autonomy
There is no "Supervisor" or "Gating Network" controlling the flow. Agents A and B operate on their own loops. Communication is treated as a metabolic cost—agents only "talk" when local uncertainty is high, effectively modeling **Global Workspace Theory**.

### 2. Heterogeneous Agents
We are not averaging identical models. The system is composed of functionally distinct clusters:
* **Cluster A (The Seer):** Specialized in **Prediction** (temporal sequence anticipation).
* **Cluster B (The Judge):** Specialized in **Classification** (feature discrimination).

### 3. Non-Backpropagation Learning
To support true real-time adaptation, DRIFT abandons global backpropagation through time (BPTT). Instead, it utilizes biologically plausible local learning rules (e.g., Hebbian, STDP, or energy-based inference) to allow the "synaptic binding" between clusters to form and dissolve dynamically.

## 🧪 Experiments

The repository is divided into specific test environments:

* **Test 01: The negotiation Game**
    * *Setup:* Agents must agree on the properties of a rapidly changing object.
    * *Metric:* Convergence time of the two agents' internal states.
* **Test 02: The Blind Guide**
    * *Setup:* Agent A sees the map but cannot move; Agent B moves but cannot see. They must communicate to navigate.
    * *Metric:* Path efficiency vs. Communication bandwidth usage.
* **Test 03: Composite Interference**
    * *Setup:* Introduce a task that requires *both* prediction and classification simultaneously.
    * *Metric:* Success rate compared to monolithic baselines.

## 📚 References & Inspiration

* **Functional Integration & Segregation** (Friston/Zeki)
* **Theory of Neuronal Group Selection / Reentry** (Edelman)
* **Predictive Coding & Free Energy Principle**
* **Emergent Communication in Multi-Agent RL**

---

*"Cells that fire together, wire together. But clusters that talk together, evolve together."*