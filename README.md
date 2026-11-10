# Fuzzy Logic-Based Diabetes Risk Assessment System

Soft Computing project, Thapar Institute of Engineering and Technology (7th semester, 2026).

A Mamdani fuzzy inference system (FIS) that estimates diabetes risk from four clinical inputs (Glucose, BMI, Age, Blood Pressure), evaluated on the Pima Indians Diabetes dataset. We also test whether a genetic algorithm (GA) can improve the hand-designed system.

**Team:** Shwetanshu, Vivek

## Dataset
Pima Indians Diabetes Database (768 patients, 34.9% diabetic). Source: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) / UCI repository.
Preprocessing: zeros in Glucose, BloodPressure and BMI are treated as missing values and replaced by the column median.

## Method
- **Inputs:** Glucose (Low / Medium / High), BMI (Normal / Overweight / Obese), Age (Young / Middle / Old), Blood Pressure (Low / Normal / High), using triangular and trapezoidal membership functions.
- **Output:** Risk on a 0 to 100 scale (Low / Medium / High), defuzzified by centroid.
- **Rule base:** 19 rules designed from standard medical thresholds.
- **Decision:** a patient is classified as diabetic when Risk exceeds a threshold, chosen on training data only.
- **GA experiments** (MATLAB `ga`): tuning membership function breakpoints (and in one variant, rule weights) to maximise training accuracy or AUC.

## Results

Main result: 5-fold cross-validation, every patient tested exactly once.

| System | Accuracy | AUC |
|---|---|---|
| Always predict "not diabetic" | 65.1% | 0.500 |
| **Hand-designed FIS (19 rules)** | **75.39%** | **0.816** |
| GA: membership functions only | 75.39% | 0.807 |
| GA: membership functions + rule weights | 74.87% | 0.810 |

Rule refinement (full dataset, before using train/test splits): 7 starter rules gave 68.23%; 19 rules with a tuned threshold gave 76.69%. This figure was tuned and measured on the same data, so it is optimistic.

Single 70/30 split (seed 1): 73.48% before GA, 75.65% after GA. Repeating over 5 random splits:

| | Test accuracy |
|---|---|
| Before GA | 77.30% +/- 2.63 |
| After GA | 76.96% +/- 1.37 |

The GA was better in only 1 of 5 splits.

**Finding:** the GA did not give a consistent improvement over the hand-designed rule base. Likely reasons: small, noisy dataset; medical-threshold rules already capture the main relationships; extra tunable parameters let the GA fit noise in the training data.

## Repository structure
```
code/       MATLAB scripts
data/       pima-indians-diabetes.csv
results/    result tables (CSV)
figures/    membership function plots, control surface, accuracy charts
archive/    early development scripts
```

| Script | Purpose |
|---|---|
| `code/diabetes_ga_v2.m` | Final experiment: hand-designed FIS vs two GA variants, 5-fold cross-validation |
| `code/diabetes_ga.m` | First GA attempt on a single 70/30 split |
| `code/diabetes_multiseed.m` | Repeats the single-split GA comparison over 5 random splits |

## How to run
1. Open MATLAB and set the Current Folder to `code/`.
2. Run `diabetes_ga_v2.m` (about 30 to 45 minutes because it runs 10 GA optimisations).
3. Results are written to `results/` and `figures/`.

**Requirements:** MATLAB R2024a with Fuzzy Logic Toolbox, Global Optimization Toolbox, and Statistics and Machine Learning Toolbox.
