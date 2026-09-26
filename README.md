# BrainScan AI: Explainable Deep Learning for Alzheimer's Disease Stage Classification Using OASIS MRI Images

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Project](https://img.shields.io/badge/Project-MSc%20Research-purple)
![Use](https://img.shields.io/badge/Use-Educational%20Only-red)

## Project Overview

**BrainScan AI** is an MSc Data Science and Artificial Intelligence research project that builds an explainable deep-learning pipeline for classifying Alzheimer's disease stages from OASIS-derived MRI image slices.

The repository contains the final notebook and project outputs generated during the research implementation:

- Final `.ipynb` notebook
- `figures/` folder containing EDA, model evaluation and Grad-CAM visualisations
- `tables/` folder containing metric tables and result summaries
- `huggingface_space/` folder containing the educational Streamlit/Hugging Face prototype files

> **Important disclaimer:** This project is for academic research and educational demonstration only. It is **not** a medical device and must not be used for clinical diagnosis, patient screening, treatment decisions or real-world healthcare decision-making.

---

## Research Question

**To what extent can explainable deep-learning models classify Alzheimer's disease stage from OASIS-derived MRI image slices, and how do dataset imbalance, model choice and Grad-CAM explainability affect interpretation?**

---

## Aim and Objectives

### Aim

To design, implement and evaluate an explainable deep-learning pipeline using dataset audit, reproducible modelling, imbalance-aware evaluation and visual explanation.

### Objectives

1. Audit the OASIS/Kaggle MRI dataset for class balance, patient/session structure, selected slice range and similarity risks.
2. Implement a reproducible CNN and transfer-learning pipeline using Python and TensorFlow/Keras.
3. Compare CompactCNN, MobileNetV2, ResNet50 and EfficientNetB0 using imbalance-aware evaluation metrics.
4. Generate Grad-CAM explanations to inspect model behaviour.
5. Package the notebook, outputs and educational prototype without making clinical diagnostic claims.

---

## Dataset

The project uses the **OASIS Alzheimer's Detection** dataset from Kaggle, derived from the Open Access Series of Imaging Studies (OASIS).

### Dataset Links

- Kaggle dataset: https://www.kaggle.com/datasets/ninadaithal/imagesoasis
- OASIS source: https://sites.wustl.edu/oasisbrains/

### Dataset Classes

| Class | Image Count | Approximate Share |
|---|---:|---:|
| Non-Demented | 67,222 | 77.77% |
| Very Mild Demented | 13,725 | 15.88% |
| Mild Demented | 5,002 | 5.79% |
| Moderate Demented | 488 | 0.56% |

The dataset is highly imbalanced, so accuracy alone is not sufficient. The project therefore uses balanced accuracy, macro-F1, ROC-AUC, precision-recall analysis, confusion matrices, calibration and error analysis.

---

## Repository Structure

```text
brainscan-ai-alzheimers-classification/
│
├── README.md
├── BrainScan_AI_Final_Implementation.ipynb
│
├── figures/
│   ├── dataset visualisations
│   ├── EDA figures
│   ├── model comparison plots
│   ├── confusion matrices
│   ├── ROC and precision-recall curves
│   ├── calibration and reliability plots
│   ├── error-analysis figures
│   └── Grad-CAM explainability outputs
│
├── tables/
│   ├── dataset summary tables
│   ├── model comparison metrics
│   ├── per-class metrics
│   └── test/evaluation outputs
│
└── huggingface_space/
    ├── app.py
    ├── requirements.txt
    ├── class_names.json
    ├── metrics.json
    ├── README.md
    └── model weights / metadata
```

### Recommended GitHub Upload

Upload these items:

1. Final `.ipynb` notebook
2. `figures/` folder
3. `tables/` folder
4. `huggingface_space/` folder
5. `README.md`

Do **not** upload the raw OASIS MRI dataset unless the dataset licence and Kaggle/OASIS terms clearly allow redistribution. Provide the dataset links instead.

---

## System Workflow

```text
OASIS/Kaggle MRI dataset
        ↓
Metadata construction
        ↓
Dataset audit and EDA
        ↓
Image preprocessing
        ↓
Train / validation / test split
        ↓
Training-set balancing
        ↓
Model training
        ↓
Model comparison
        ↓
Untouched test-set evaluation
        ↓
Grad-CAM explainability
        ↓
Export figures, tables and prototype files
```

---

## Preprocessing

The preprocessing pipeline includes:

1. **Image loading** — MRI image paths are loaded from class folders.
2. **Label encoding** — class names are converted into numeric labels.
3. **Image resizing** — all images are resized to a fixed input size for CNN models.
4. **Colour-channel preparation** — grayscale MRI slices are prepared in a model-compatible format.
5. **Normalisation / scaling** — pixel values are scaled according to model requirements.
6. **Metadata construction** — file path, class, class index, slice information and available patient/session information are organised into structured tables.
7. **Dataset audit** — class balance, image statistics, z-axis slices and similarity risks are inspected.
8. **Training-set balancing** — only the training set is balanced; validation and test sets remain untouched.
9. **Safe augmentation** — light augmentation is used where suitable, while heavy transformations are avoided because MRI anatomy should not be distorted.

---

## Models Implemented

### CompactCNN

A custom convolutional neural network used as a baseline. It learns spatial features directly from MRI slices using convolution, pooling, batch normalisation and dense layers.

### MobileNetV2

A lightweight transfer-learning model using inverted residual blocks. It was selected as the final model because it achieved the strongest macro-F1 score while remaining efficient for prototype packaging.

### ResNet50

A deeper transfer-learning model using residual skip connections. It was included to test whether a deeper CNN architecture improves MRI feature extraction.

### EfficientNetB0

A transfer-learning model using compound scaling. It was included to compare a modern efficient architecture against MobileNetV2 and ResNet50.

---

## Final Model Performance

The final selected model was **MobileNetV2**.

| Metric | MobileNetV2 Result |
|---|---:|
| Accuracy | 67.31% |
| Balanced Accuracy | 82.16% |
| Macro-F1 | 60.83% |
| Top-2 Accuracy | 90.27% |
| Macro ROC-AUC | 92.61% |
| Expected Calibration Error | 0.144 |

MobileNetV2 was selected because it achieved the strongest macro-F1 score among the tested models. Macro-F1 was important because the dataset is heavily imbalanced and accuracy alone could hide weak minority-class performance.

---

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct predictions |
| Balanced Accuracy | Gives equal importance to each class |
| Precision | Shows how reliable positive predictions are |
| Recall / Sensitivity | Shows how many true cases are detected |
| Macro-F1 | Balances precision and recall across all classes |
| ROC-AUC | Measures one-vs-rest discrimination ability |
| Precision-Recall Curves | Useful for imbalanced datasets |
| Confusion Matrix | Shows class-wise correct and incorrect predictions |
| Expected Calibration Error | Checks whether model confidence is reliable |
| Ordinal Severity Distance | Measures how far wrong predictions are across disease-stage order |

---

## Explainability with Grad-CAM

Grad-CAM was used to visualise image regions that influenced the final model's predictions.

Grad-CAM helps inspect whether the model is focusing on meaningful image regions or irrelevant artefacts. However, Grad-CAM is **not clinical proof** and does not replace expert medical interpretation.

---

## Key Output Folders

### `figures/`

This folder contains visual evidence such as:

- Dataset class-distribution chart
- Dataset class-share chart
- Log-scale imbalance chart
- Patient/session and slice audit figures
- Random MRI examples
- Z-axis slice progression examples
- Image-statistics plots
- Correlation heatmap
- Training and validation curves
- Model comparison charts
- Per-class precision, recall and F1 heatmaps
- Confusion matrices
- ROC and precision-recall curves
- Reliability diagram
- High-confidence misclassification examples
- Grad-CAM heatmaps and overlays

### `tables/`

This folder contains tabular outputs such as:

- Dataset summary tables
- Train/validation/test split summary
- Model comparison metrics
- Per-class precision, recall and F1 results
- Final selected-model metrics
- Testing/evaluation evidence

### `huggingface_space/`

This folder contains the educational prototype files. Typical files include:

- `app.py`
- `requirements.txt`
- `class_names.json`
- `metrics.json`
- model weights or metadata
- prototype support files

The prototype must include a clear disclaimer that predictions are not clinical diagnoses.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Chandana162002/brainscan-ai-alzheimers-classification.git
cd brainscan-ai-alzheimers-classification
```

Install dependencies:

```bash
pip install -r huggingface_space/requirements.txt
```

If you create a root-level `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

---

## How to Run the Notebook

### Kaggle

1. Open Kaggle Notebooks.
2. Add the OASIS Alzheimer's Detection dataset.
3. Upload or open the final `.ipynb` notebook.
4. Confirm that the dataset path is correct.
5. Run all cells from top to bottom.
6. Review outputs in `figures/`, `tables/` and `huggingface_space/`.

Example Kaggle dataset path:

```python
DATA_DIR = "/kaggle/input/imagesoasis/Data"
```

### Local System

1. Download the dataset from Kaggle.
2. Extract it locally.
3. Update the dataset path inside the notebook.
4. Run the notebook using Jupyter Notebook, JupyterLab or VS Code.

Example local path:

```python
DATA_DIR = "path/to/imagesoasis/Data"
```

---

## How to Run the Prototype

```bash
cd huggingface_space
streamlit run app.py
```

The prototype accepts an uploaded MRI image and returns class probabilities. The result should be interpreted only as an educational AI output.

---

## Ethics and Data Governance

This project follows a **UREC1 no-human-participants route**.

The project does not involve:

- interviews
- surveys
- observations
- new patient scans
- clinical recruitment
- identifiable DICOM metadata
- direct personal-data collection
- real patient decision-making

Only public, de-identified secondary MRI image data is used. Raw MRI image files are not redistributed through this repository unless allowed by the original dataset terms.

---

## Limitations

1. The dataset is slice-level, so multiple images may come from the same patient or scan session.
2. The class distribution is highly imbalanced.
3. The project uses processed 2D JPEG images rather than full 3D MRI volumes.
4. The results are image-level benchmark results, not independent patient-level clinical validation.
5. Model confidence is not perfectly calibrated.
6. Grad-CAM heatmaps improve transparency but do not prove clinical correctness.
7. The system is not suitable for diagnosis or real healthcare decision-making.

---

## Future Work

Future improvements could include:

1. Patient-level splitting.
2. Independent external validation.
3. Larger and more balanced multi-site MRI datasets.
4. 3D CNN or vision-transformer comparison.
5. Stronger calibration methods.
6. Better uncertainty estimation.
7. Expert review after amended ethics approval.
8. Improved deployment testing and monitoring.

---

## AI Use Transparency

AI tools supported wording refinement, structure planning and debugging explanation. Experimental results, dataset access, notebook execution, model training, metrics, figures and final interpretation were independently produced and checked by the project author.

No dataset, metric, participant feedback or experimental result was fabricated.

---

## Acknowledgements

I thank **Copley Royce** for guidance on project scope, UREC1 ethics boundary, methodology and final evaluation focus.

I am grateful to **Sheffield Hallam University** and the **College of Business, Technology and Engineering** for the research project framework, handbook and group-meeting guidance.

I acknowledge the **OASIS investigators** and **Kaggle/OASIS contributors** for providing public, de-identified MRI data used in this secondary-data project.

---

## References

Aithal, N. (n.d.). *OASIS Alzheimer's detection* [Data set]. Kaggle.  
https://www.kaggle.com/datasets/ninadaithal/imagesoasis

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770–778.  
https://doi.org/10.1109/CVPR.2016.90

Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle-aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience, 19*(9), 1498–1507.  
https://doi.org/10.1162/jocn.2007.19.9.1498

Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). Open Access Series of Imaging Studies: Longitudinal MRI data in nondemented and demented older adults. *Journal of Cognitive Neuroscience, 22*(12), 2677–2684.  
https://doi.org/10.1162/jocn.2009.21407

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4510–4520.  
https://doi.org/10.1109/CVPR.2018.00474

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision*, 618–626.  
https://doi.org/10.1109/ICCV.2017.74

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning, 97*, 6105–6114.  
https://proceedings.mlr.press/v97/tan19a.html

World Health Organization. (2023). Dementia.  
https://www.who.int/news-room/fact-sheets/detail/dementia

---

## Responsible Use Statement

This repository is for academic research, learning and demonstration only. Users must follow the original OASIS and Kaggle dataset terms. The project is not approved for clinical deployment and should not be used to make healthcare decisions.
