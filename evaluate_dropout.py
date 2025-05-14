# evaluate_dropout.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import lasagne
from classifier import train_model
from mlLeaks import readCIFAR10, preprocessingCIFAR, shuffleAndSplitData
import os

# SETTINGS
dataset_path = './data/cifar-10-batches-py-official'
dropout_settings = [0.0, 0.25, 0.5, 0.75]  # no dropout -> high dropout
results = {
    'dropout': [],
    'precision': [],
    'recall': [],
    'accuracy': []
}
# print("checkpoint 2")

# load & preprocess
print("loading CIFAR-10 dataset")
dataX, dataY, _, _ = readCIFAR10(dataset_path)
cluster = 10520
trainX, trainY, _, _, testX, testY, _, _ = shuffleAndSplitData(dataX, dataY, cluster)
trainX, testX = preprocessingCIFAR(trainX, testX)


# evaluate each dropout setting
for d in dropout_settings:
    print(f"\nevaluating with dropout={d}")
    dataset = (trainX.astype(np.float32), trainY.astype(np.int32), testX.astype(np.float32), testY.astype(np.int32))
    output_layer = train_model(dataset, model='cnn', dropout_p=d, epochs=50)
    # print("checkpoint 4 for d=", d)

    # get predictions
    input_var = lasagne.layers.get_all_layers(output_layer)[0].input_var
    prediction_fn = lasagne.layers.get_output(output_layer, deterministic=True)
    test_fn = lasagne.theano.function([input_var], prediction_fn)

    preds = []
    for i in range(0, len(testX), 100):
        batch = testX[i:i+100]
        probs = test_fn(batch)
        preds.extend(np.argmax(probs, axis=1))
        # print("checkpoint 6 for d=", d, "batch=", i)

    preds = np.array(preds)
    acc = accuracy_score(testY, preds)
    prec = precision_score(testY, preds, average='macro', zero_division=0)
    rec = recall_score(testY, preds, average='macro', zero_division=0)

    results['dropout'].append(d)
    results['accuracy'].append(acc)
    results['precision'].append(prec)
    results['recall'].append(rec)
    # print("checkpoint 7 for d=", d)

# print("checkpoint 8 outside loop")
# plotting!
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(results['dropout'], results['precision'], marker='o', label='Precision')
plt.plot(results['dropout'], results['recall'], marker='s', label='Recall')
plt.title('Precision and Recall vs. Dropout Ratio')
plt.xlabel('Dropout Ratio')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend()
# print("checkpoint 9")

plt.subplot(1, 2, 2)
plt.plot(results['dropout'], results['accuracy'], marker='^', color='green', label='Accuracy')
plt.title('Accuracy vs. Dropout Ratio')
plt.xlabel('Dropout Ratio')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
# print("checkpoint 10")

plt.tight_layout()
plt.savefig('dropout_evaluation_results.png')
plt.show()
# print("checkpoint 11")

# save results to file
np.savez('dropout_eval_metrics.npz', **results)
print("\nEvaluation complete - results saved to 'dropout_eval_metrics.npz' and plotted in 'dropout_evaluation_results.png'.")
