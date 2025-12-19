# DRIFT: Decentralized Real-time Integration & Functional Transfer

> **Status:** Experimental / Active Research
> **Paradigm:** Bio-Mimetic AI / Online Learning
> **Learning Rule:** Decoupled Backpropagation / Differentiable Chaining

## 🧠 The Philosophy

**DRIFT** is an experimental framework exploring the boundaries of **Decentralized Functional Integration**.

Current AI architectures rely heavily on monolithic networks where the entire system must update simultaneously. While powerful, this fails to mimic the modular, emergent nature of biological intelligence. In the brain, specialized regions (e.g., visual cortex vs. prefrontal cortex) operate independently but bind their information together dynamically to solve complex tasks.

**The core hypothesis of DRIFT is:**
*Two specialized, heterogeneous neural agents (e.g., a Predictor and a Classifier) can autonomously develop a communication protocol in real-time. By utilizing a targeted backpropagation step only at the point of interaction, they can "chain" their capabilities to solve a composite task that neither can solve alone, without merging into a single rigid model.*

## 🎯 Project Goals

The primary goal is to demonstrate **Synergistic Emergence**: where the combined capability of communicating agents is qualitatively higher than the sum of their individual parts.

We are conducting a series of tests to validate:
1.  **Emergent Communication:** Can two agents invent a signaling method to share latent states?
2.  **Task Binding:** Can a "Predictor" and a "Classifier" negotiate a consensus on reality to solve a third, unseen task?
3.  **Online Plasticity:** Can this negotiation happen in *real-time* (online learning) by passing gradients across the communication link?

## ⚙️ Architecture & Methodology

DRIFT differs from standard Mixture of Experts (MoE) or GANs in three critical ways:

### 1. Decentralized Autonomy
There is no "Supervisor" or "Gating Network" controlling the flow. Agents A and B operate on their own loops. Communication is treated as a metabolic cost—agents only "talk" when local uncertainty is high, effectively modeling **Global Workspace Theory**.

### 2. Heterogeneous Agents
We are not averaging identical models. The system is composed of functionally distinct clusters:
* **Cluster A (The Seer):** Specialized in **Prediction** (temporal sequence anticipation).
* **Cluster B (The Judge):** Specialized in **Classification** (feature discrimination).

### 3. Differentiable Chaining
To support true real-time adaptation, DRIFT utilizes a **Decoupled Backpropagation** strategy.
* The networks perform standard backpropagation for their specific local tasks.
* When they interact, a specific "bridging" gradient is calculated and passed between them.
* This allows the "communication channel" itself to be optimized mathematically via a **backprop step during chaining**, enabling the "synaptic binding" between clusters to form and dissolve dynamically without freezing the weights.

## 🧪 Experiments

The repository is divided into specific test environments:

* **Test 01: The Negotiation Game**
    * *Setup:* Agents must agree on the properties of a rapidly changing object.
    * *Metric:* Convergence time of the two agents' internal states via the chaining step.
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