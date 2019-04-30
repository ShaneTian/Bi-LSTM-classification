# Bi-LSTM classification
Bi-LSTM classification by TensorFlow 2.0.0 ( tf.keras mainly ).
The data of this project is same as [TextCNN](https://github.com/ShaneTian/TextCNN).
## Software environments
- tensorflow-gpu 2.0.0-alpha0
- python 3.6.7
- pandas 0.24.2
- numpy 1.16.2
- scikit-learn 0.20.3
- jieba 0.39

## Data
- Vocabulary size: 35385
- Number of classes: 18
- Train/Test split: 20351/2261

## Model architecture
```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_data (InputLayer)      [(None, 150)]             0
_________________________________________________________________
embedding (Embedding)        (None, 150, 512)          18117632
_________________________________________________________________
masking (Masking)            (None, 150, 512)          0
_________________________________________________________________
Bi-LSTM (Bidirectional)      (None, 200)               490400
_________________________________________________________________
dense (Dense)                (None, 18)                3618
=================================================================
Total params: 18,611,650
Trainable params: 18,611,650
Non-trainable params: 0
_________________________________________________________________
```
## Model parameters
- Padding size: 150
- Embedding dim: 512
- LSTM dim: 100
- Dropout rate: 0.4
- Regularizers lambda: 0.001
- Batch size: 64
- Epochs: 15
- Fraction validation: 0.05 (1018 samples)
- Total parameters: 18,611,650

## Run
### Train result
Use 20351 samples after 15 epochs:

| Loss | Accuracy | Val loss | Val accuracy |
| --- | --- | --- | --- |
| 0.0783 | 0.9888 | 0.3638 | 0.9194 |
### Test result
Use 2261 samples:

| Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
| --- | --- | --- | --- |
| 0.9341 | 0.9351 | 0.9323 | **0.9328** |
### Images
#### Accuracy
![Accuracy](https://github.com/ShaneTian/Bi-LSTM-classification/blob/master/results/2019-04-29-15-56-39/acc.jpg)

#### Loss
![Loss](https://github.com/ShaneTian/Bi-LSTM-classification/blob/master/results/2019-04-29-15-56-39/loss.jpg)

#### Confusion matrix
![Confusion matrix](https://github.com/ShaneTian/Bi-LSTM-classification/blob/master/results/2019-04-29-15-56-39/confusion_matrix.jpg)

### Usage
```
usage: train.py [-h] [-t TEST_SAMPLE_PERCENTAGE] [-p PADDING_SIZE]
                [-e EMBED_SIZE] [-d DROPOUT_RATE] [-c NUM_CLASSES]
                [-l REGULARIZERS_LAMBDA] [-b BATCH_SIZE] [--epochs EPOCHS]
                [--fraction_validation FRACTION_VALIDATION]
                [--results_dir RESULTS_DIR]

This is the Bi-LSTM train project.

optional arguments:
  -h, --help            show this help message and exit
  -t TEST_SAMPLE_PERCENTAGE, --test_sample_percentage TEST_SAMPLE_PERCENTAGE
                        The fraction of test data.(default=0.1)
  -p PADDING_SIZE, --padding_size PADDING_SIZE
                        Padding size of sentences.(default=150)
  -e EMBED_SIZE, --embed_size EMBED_SIZE
                        Word embedding size.(default=512)
  -d DROPOUT_RATE, --dropout_rate DROPOUT_RATE
                        Dropout rate in softmax layer.(default=0.4)
  -c NUM_CLASSES, --num_classes NUM_CLASSES
                        Number of target classes.(default=18)
  -l REGULARIZERS_LAMBDA, --regularizers_lambda REGULARIZERS_LAMBDA
                        L2 regulation parameter.(default=0.001)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Mini-Batch size.(default=64)
  --epochs EPOCHS       Number of epochs.(default=15)
  --fraction_validation FRACTION_VALIDATION
                        The fraction of validation.(default=0.05)
  --results_dir RESULTS_DIR
                        The results dir including log, model, vocabulary and
                        some images.(default=./results/)
```

```
usage: test.py [-h] [-p PADDING_SIZE] [-c NUM_CLASSES] results_dir

This is the Bi-LSTM test project.

positional arguments:
  results_dir           The results dir including log, model, vocabulary and
                        some images.

optional arguments:
  -h, --help            show this help message and exit
  -p PADDING_SIZE, --padding_size PADDING_SIZE
                        Padding size of sentences.(default=150)
  -c NUM_CLASSES, --num_classes NUM_CLASSES
                        Number of target classes.(default=18)
```
#### You need to know...
1. You need to alter `load_data_and_write_to_file` function in `data_helper.py` to match you data file;
2. This code did not use embedding vector, you can use it, maybe it is greater;
3. The model is saved by `hdf5` file;
4. Tensorboard is available.
