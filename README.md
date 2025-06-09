# NeuralNetworkAnalysis

## Neural Network-Based Credit Risk Classification

### Modeling Creditworthiness Using German Credit Data
This project builds and evaluates multiple neural network architectures to predict loan default risk using the German Credit dataset. It focuses not just on prediction accuracy, but also on model reliability through stability, and business impact through lift—a rare but crucial metric in credit modeling and financial applications.
This work reflects the design of deployable, risk-sensitive machine learning systems—prioritizing interpretability and robustness over raw accuracy.

## Objective
The goal is to design, train, and evaluate a range of neural networks with varying:
Optimizers (ADAM, SGD)
Learning rate schedules (step decay, exponential decay, ReduceLROnPlateau)
Regularization techniques (L2 penalties, EarlyStopping)
Batch sizes and epoch lengths
Loss function: Binary Crossentropy
Models were tested and validated using stratified sampling (85% training, 15% test split), and a Hyperband Tuner was used to identify the optimal learning rate. The final selection criteria incorporated accuracy, model lift, and stability based on visual response consistency.

## Dataset: German Credit Data
Target Variable: GOOD_BAD
Event of Interest: BAD (credit defaulter)
Type: Tabular classification, ~1000 rows
Encoding: Categorical variables handled internally; one-hot encoding not required.

## Evaluation Framework
Metric	Description	Weight in Score	Threshold
Accuracy	Overall classification performance	45%	> 0.60
Lift	Cumulative lift at 10th percentile	30%	> 0.25
Stability	Visual inspection of lift curve smoothness (binary score)	25%	Must be stable (1)
Composite Score = 0.45 × Accuracy + 0.30 × Lift + 0.25 × Stability

## Models Compared
| Model        | Accuracy | Lift   | Stability | Score |
|--------------|----------|--------|-----------|-------|
| ADAM_32_SD   | 0.740    | 0.476  | ✅         | ⭐️ Best |
| ADAM_32_RP   | 0.773    | 0.524  | ❌         |       |
| ADAM_64_01   | 0.760    | 0.429  | ✅         |       |
| SGD_32_RP    | 0.760    | 0.429  | ✅         |       |

## Best Model: ADAM_32_SD
Accuracy: 0.740
Lift: 0.476
Stability: Stable
Composite Score: 0.345
Although ADAM_32_RP achieved slightly higher raw accuracy and lift, it lacked stability across validation folds—indicating overfitting risk. ADAM_32_SD emerged as the most balanced and deployable model.

## Why This Matters (Real-World Relevance)
In financial systems and policy settings, a performant model isn’t enough—model stability and actionable insight at high-risk deciles matter more. This project simulates:
A credit scoring pipeline
With risk-based evaluation metrics
Focused on model trustworthiness and generalization
This is critical for applications in:
Microfinance loan approvals
Consumer credit risk forecasting
Ethical algorithmic decision-making
