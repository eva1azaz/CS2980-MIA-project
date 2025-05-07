The code for the ML-Leaks paper. All three attacks are implmented using the CIFAR-10 and News datasets.

My project uses the CIFAR-10 dataset. Download the official binary version from: 
https://www.cs.toronto.edu/~kriz/cifar.html  
After downloading, extract the contents and place them in: ./data/cifar-10-batches-py-official

To switch between base model and dropout: 
Comment out whichever model is not in use at the top of `deeplearning.py` where:
    A. classifier: base model
    B. dropout_classifier: dropout model 