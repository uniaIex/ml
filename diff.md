# Understanding the Differences: AI vs. Machine Learning vs. Neural Networks

## 1. **Artificial Intelligence (AI)**  
**Definition**:  
The broad field of creating machines or systems capable of performing tasks that typically require human intelligence.  

**Key Characteristics**:  
- **Goal**: Mimic human-like reasoning, decision-making, or creativity.  
- **Scope**: Includes both rule-based systems and learning-based approaches.  
- **Examples**:  
  - Rule-based: Chess-playing engines, basic chatbots.  
  - Learning-based: Self-driving cars, voice assistants.  

**Subfields**:  
- Machine Learning  
- Computer Vision  
- Natural Language Processing (NLP)  
- Robotics  

---

## 2. **Machine Learning (ML)**  
**Definition**:  
A subset of AI focused on developing algorithms that learn patterns from data *without explicit programming*.  

**Key Characteristics**:  
- **Data-Driven**: Improves performance with more data.  
- **Types**:  
  - Supervised Learning (labeled data)  
  - Unsupervised Learning (unlabeled data)  
  - Reinforcement Learning (reward-based learning)  
- **Examples**:  
  - Email spam filters.  
  - Predictive text suggestions.  

**Common Algorithms**:  
- Decision Trees  
- Support Vector Machines (SVM)  
- **Neural Networks** (see below)  

---

## 3. **Neural Networks (NN)**  
**Definition**:  
A specific class of machine learning algorithms inspired by the human brain’s structure, using interconnected nodes ("neurons") to process data.  

**Key Characteristics**:  
- **Structure**: Layers of neurons (input, hidden, output).  
- **Deep Learning**: A subset using **deep neural networks** (many hidden layers).  
- **Examples**:  
  - Image recognition (e.g., identifying cats in photos).  
  - Speech-to-text systems.  

**Types**:  
- Convolutional Neural Networks (CNNs) for images.  
- Recurrent Neural Networks (RNNs) for sequences (e.g., text).  
- Transformers (e.g., GPT-4 for language tasks).  

---

## Relationship Hierarchy  
```mermaid
graph TD
  A[Artificial Intelligence] --> B[Machine Learning]
  A --> C[Rule-Based Systems]
  B --> D[Neural Networks]
  D --> E[Deep Learning]
  B --> F[Other ML Algorithms<br>e.g., Decision Trees, SVM]
  

## Key Differences Table

| Aspect                  | Artificial Intelligence (AI)                                                                 | Machine Learning (ML)                                                                 | Neural Networks (NN)                                                                 |
|-------------------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| **Scope**               | Broadest field (encompasses all systems mimicking human intelligence)                       | Subset of AI focused on learning patterns from data                                   | Subset of ML using brain-inspired algorithms                                         |
| **Function**            | Solves tasks requiring reasoning, decision-making, or creativity                            | Trains models to make predictions or decisions using data                            | Processes data through interconnected neuron layers                                  |
| **Dependency**          | Can exist without ML (e.g., rule-based systems)                                              | Requires labeled or unlabeled data for training                                       | Requires large datasets, computational power, and layered architectures              |
| **Example**             | Autonomous robot navigating a room                                                          | Predicting stock prices using historical trends                                       | Generating realistic images with Generative Adversarial Networks (GANs)              |
| **Learning Mechanism**  | May use pre-programmed rules or ML                                                           | Uses statistical algorithms (e.g., regression, clustering)                           | Uses forward/backward propagation to adjust neuron weights                           |
| **Complexity**          | Can be simple (e.g., rule-based chatbots) or highly complex (e.g., autonomous systems)       | Moderate complexity (depends on algorithm and data size)                             | High complexity (deep learning models with millions of parameters)                   |

---

### Hierarchy Recap:
- **AI**: The overarching concept (e.g., self-driving cars).  
- **ML**: A toolset within AI for learning from data (e.g., training a recommendation system).  
- **NN**: A specialized ML technique for complex pattern recognition (e.g., facial recognition models).  


Tensors ->
    "A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes." (https://www.tensorflow.org/guide/tensor)

    It should't surprise you that tensors are a fundemental apsect of TensorFlow. They are the main objects that are passed around and manipluated throughout the program. Each tensor represents a partialy defined computation that will eventually produce a value. TensorFlow programs work by building a graph of Tensor objects that details how tensors are related. Running different parts of the graph allow results to be generated.

    Each tensor has a data type and a shape.

    Data Types Include: float32, int32, string and others.

    Shape: Represents the dimension of data.

    Just like vectors and matrices tensors can have operations applied to them like addition, subtraction, dot product, cross product etc.


