The code for the ML-Leaks paper. All three attacks are implmented using the CIFAR-10 and News datasets.

To switch between base model and dropout: 
Comment out whichever model is not in use at the top of `deeplearning.py` where:
    A. classifier: base model
    B. dropout_classifier: dropout model 