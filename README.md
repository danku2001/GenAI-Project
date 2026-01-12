# HelmetGuard  
## Safety Classification of Construction Site Images Using Synthetic and Real Data

---

## Project Motivation
Construction sites are high-risk environments, where accidents often occur due to workers not following safety regulations. One of the most basic safety requirements is wearing a protective helmet, yet in practice this rule is not always followed.

In real construction sites, safety officers cannot continuously monitor multiple cameras and areas at the same time. This project explores whether an image-based machine learning system can automatically classify construction site images as **safe** or **unsafe** based on helmet usage.

---

## Problem Definition
The task is formulated as a supervised binary image classification problem.

**Input:**  
An RGB image taken at a construction site.

**Output:**  
A single label:
- **Safe** – all visible workers are wearing helmets  
- **Unsafe** – at least one visible worker is not wearing a helmet  

The model makes a global decision for the entire image, without localizing individual workers.

---

## Dataset
Due to the limited availability of labeled real-world construction images, the dataset combines **synthetic** and **real** data.

### Synthetic Data
- Approximately **1,020 images** generated using a Stable Diffusion-based pipeline  
- Images represent construction site scenes with workers wearing helmets or without helmets  
- Text prompts were carefully designed to describe safe and unsafe scenarios  
- Negative prompts were used to reduce unrealistic artifacts  
- All generated images were manually reviewed and cleaned  
- Images were labeled as **safe** or **unsafe** based on helmet usage  

### Real Data
- Collected from public web sources  
- **40 safe images** and **19 unsafe images**  
- Split into training and test sets  
- Unsafe samples were oversampled during training to reduce class imbalance  

---

## Synthetic Data Generation
Synthetic images were generated using a diffusion-based generative model (Stable Diffusion) executed in a Colab environment. Prompts explicitly described construction workers, helmets, and site environments, while negative prompts were used to avoid unrealistic outputs.

After generation, all images were manually inspected to ensure visual quality and correct labeling. This cleaning step was necessary to reduce noise and labeling errors in the synthetic dataset.

---

## Model and Training
- **Backbone:** ResNet18 pretrained on ImageNet  
- **Classifier:** 2-class classification head  
- **Input size:** 224×224  
- **Loss function:** Cross-entropy  
- **Optimizer:** Adam (learning rate = 1e-4)  
- **Data augmentation:** random crops, horizontal flips, color jitter  

---

## Experiments
Three experimental setups were evaluated:

1. **Baseline**  
   ResNet18 trained only on synthetic data  

2. **Synthetic + Real Data**  
   Training on synthetic data with a small number of real images added  

3. **Synthetic + Real Data with Oversampling**  
   Same as experiment 2, with oversampling applied to unsafe real samples  

---

## Results
The synthetic-only model performs well on synthetic validation data but generalizes poorly to real-world images. Adding a small number of real training samples significantly improves real test performance and reduces class imbalance. Oversampling unsafe examples further improves the balance between safe and unsafe predictions.

Overall, the experiments demonstrate that synthetic data is useful for bootstrapping, but real data is essential for deployment in real-world scenarios.

---

## Discussion and Limitations
- There is a noticeable domain gap between synthetic and real images  
- The real dataset is relatively small and limits generalization  
- The model performs scene-level classification and does not localize individual workers  
- False positives and false negatives present a tradeoff between safety and usability  

---

For dataset access details, see the `data/` directory.

---

## Experiments & Results

We evaluate our models on the real test set (30 safe, 14 unsafe).
The main metric is accuracy, and we also report per-class accuracy for `safe` and `unsafe`.

| Experiment | Model    | Train data                                        | Real acc | Safe acc | Unsafe acc |
|-----------|----------|---------------------------------------------------|---------:|---------:|-----------:|
| 1         | ResNet18 | Synthetic only                                    | 31.5%    | 12.5%    | 85.7%      |
| 2         | ResNet18 | Synth + few real (10 safe, 5 unsafe)              | 63.6%    | 76.7%    | 35.7%      |
| 3         | ResNet18 | Synth + real (10 safe, 25 unsafe, oversampled)    | 52.3%    | 56.7%    | 42.9%      |
| 4         | ViT-B/16 | Synth + real (10 safe, 25 unsafe, oversampled)    | 63.6%    | 73.3%    | 42.9%      |


Overall, synthetic-only training performs well on synthetic images but does not generalize to real safe images.
Adding a small amount of real data and oversampling unsafe examples strongly changes the model's behaviour and
leads to a more balanced trade-off between safe and unsafe performance.

