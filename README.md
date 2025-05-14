
# Defending Against Membership Inference Attacks

Eva Azazoglu, Brown University CSCI2980 Research Project, May 2025.

This repository extends the [ML-Leaks framework](https://github.com/AhmedSalem2/ML-Leaks) to evaluate and compare defenses against membership inference attacks (MIAs), including:
- **Dropout** (model-level defense)
- **Label smoothing** (model-level defense)
- **Confidence score masking** (output-level defense)

The project focuses on the CIFAR-10 dataset and implements all defense methods both individually and in combination.

## Key Resources
- [ML-Leaks Paper (Salem et al., NDSS 2019)](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_05A-2_Salem_paper.pdf)
- [Original GitHub Repo](https://github.com/AhmedSalem2/ML-Leaks)
- [This Repo (Eva Azazoglu)](https://github.com/eva1azaz/CS2980-MIA-project)

---

## Project Structure

├── data/
│ ├── cifar-10-batches-py-official/ # CIFAR-10 dataset (downloaded)
│ └── CIFAR10/
│ ├── attackerModelData/ # Saved shadow/target outputs
│ ├── Preprocessed/ # Processed model I/O
│ └── model/CIFAR10/ # Saved trained models
├── classifier.py # Base CNN classifier
├── dropout_classifier.py # CNN with dropout
├── smoothing_classifier.py # CNN with label smoothing
├── dropout_smoothing_classifier.py # CNN with dropout + smoothing
├── evaluate_dropout.py # Dropout effect analysis script
├── deeplearning.py # Target model training + attack data
├── mlLeaks.py # End-to-end experiment pipeline
├── README.md

---

## Setup

To reate and configure the environment:

```bash
conda create -n mlleaks-env python=3.8 -y
conda activate mlleaks-env
conda install -c conda-forge numpy=1.23.5 scipy=1.10.1 scikit-learn=1.3.2 matplotlib
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
pip install Theano==1.0.5
```

To download and extract CIFAR-10:
```bash
mkdir -p ./data
cd ./data
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
mv cifar-10-batches-py cifar-10-batches-py-official
cd ..
```

## How to Run Experiments

### Step 1: Select Classifier Variant

Modify the import in `deeplearning.py` at the top:

```python
# For base model:
from classifier import get_cnn_model

# For dropout:
from dropout_classifier import get_cnn_model

# For label smoothing:
from smoothing_classifier import get_cnn_model

# For dropout + smoothing:
from dropout_smoothing_classifier import get_cnn_model
```

### Step 2: Run Experiments

| Defenses Used         | Dropout | Smoothing | Masking | Command                                                                 |
|-----------------------|---------|-----------|---------|-------------------------------------------------------------------------|
| None (Benchmark)      | No       | No         | No       | `python mlLeaks.py --preprocessData --trainTargetModel --trainShadowModel` |
| Dropout               | Yes      | No         | No       | *(switch to `dropout_classifier.py`)*                                  |
| Smoothing             | No       | Yes        | No       | *(switch to `smoothing_classifier.py`)*                                 |
| Masking               | No       | No         | Yes      | `python mlLeaks.py --preprocessData --trainTargetModel --trainShadowModel --maskConfidences` |
| Dropout + Masking     | Yes      | No         | Yes      | *(switch to `dropout_classifier.py`)* + `--maskConfidences`            |
| Dropout + Smoothing   | Yes      | Yes        | No       | *(switch to `dropout_smoothing_classifier.py`)*                         |
| Smoothing + Masking   | No       | Yes        | Yes      | *(switch to `smoothing_classifier.py`)* + `--maskConfidences`          |
| All Three             | Yes      | Yes        | Yes      | *(switch to `dropout_smoothing_classifier.py`)* + `--maskConfidences`  |

---

## Evaluating Dropout Effects

Use `evaluate_dropout.py` to test the impact of different dropout rates:

```bash
python evaluate_dropout.py
```

This will:

- Train the model with dropout rates: 0.0, 0.25, 0.5, 0.75  
- Save metrics in `dropout_eval_metrics.npz`  
- Plot results in `dropout_evaluation_results.png`  

## Interpreting Output

During training, you'll see model logs such as:

```yaml
Epoch 40, train loss 0.199
Testing Accuracy: 0.60
```

More detailed results:

```text
  precision    recall  f1-score   support
      ...       ...      ...       ...
```

Interpretation:

- **Target Accuracy** shows model utility (e.g. 60%)  
- **Attack Accuracy** shows MIA success (lower is better for privacy)  
- **Recall class 1 = members** (how well attacker detects members)  
- **Recall class 0 = non-members**  
- **Balanced Accuracy** = average of the two recall values  

Example:

> High recall for class 1 and low for class 0 means the attacker performs very well on members only, revealing a generalization gap.

- Dropout may reduce this gap but can harm target accuracy.  
- Smoothing often provides strong privacy with minimal utility loss.  

---

## Notes

- All models trained for **50 epochs** with **batch size 100** and **learning rate 0.01**.  
- Each defense combination saves `.npz` outputs in `data/CIFAR10/` for reuse.  
- **Confidence masking** is applied post-training and does **not** require retraining models.  

---

## Citation

If you use or build on this project, please cite:

```bibtex
@inproceedings{salem2019mlleaks,
  author = {Ahmed Salem and Yang Zhang and Mathias Humbert and Pascal Berrang and Mario Fritz and Michael Backes},
  title = {ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models},
  booktitle = {Proceedings of the Network and Distributed System Security Symposium (NDSS)},
  year = {2019},
  address = {San Diego, CA, USA},
  doi = {10.14722/ndss.2019.23119}
}
```


