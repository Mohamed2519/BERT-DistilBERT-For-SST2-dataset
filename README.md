# Text classification on SST2 Dataset
This Repo make Text classification on `SST2` dataset using various text models such as `RNN` , `LSTM` , `BERT` , `DistilBERT`

Also contain Evaluation with various evaluation metrics:
`Accuracy` `confusion matrix` `Precision` `Recall` `F1 score`  `ROC/AUC curves`
## Notebook folder contain 3 notebooks
### 1- RNN_CLS.ipynb
### 2- Lstm_CLS.ipynb
### 3- BERT&Distil_BERT_For_SST2.ipynb
with help from 
```https://github.com/YJiangcm/SST-2-sentiment-analysis```




# Clothing Articles Classification 
Clothes classification using various models. 

In this Repo, I Use Famous deepleaning architectures for image classification such as  `VGG16`, `Densenet121`,and `Efficientnetb0`.
 Training on 
 [Fashion Product Images (Small)Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)


Calculate FLOPS and MACCs for each model using `flopth` library
## Models
**`VGG16`** is a popular deep neural network architecture for image classification. It consists of 16 convolutional layers. It is known for its depth, simplicity, and strong performance in image classification tasks.
![picture](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

------------------------------------


**`DenseNet-121`** consists of multiple dense blocks, where each layer is connected to every other layer within the same block. This dense connectivity promotes feature reuse and helps in combating the vanishing gradient problem. DenseNet-121 also incorporates bottleneck layers and transition layers to reduce the number of parameters and control the spatial dimensions of the feature maps. It has been widely used for image classification tasks and has shown strong performance. DenseNet-121 is known for its efficient memory utilization and ability to capture intricate feature representations in deep networks.
![picture](https://miro.medium.com/v2/resize:fit:678/1*u4hyohOF9SIRRLBAzqYXfQ.jpeg)

------------------------------------



**`EfficientNet-B0`** is a compact and efficient convolutional neural network architecture that achieves a good balance between model size and performance. It is known for its efficiency in terms of parameter size and computational cost while still delivering competitive accuracy.

![picture](https://wisdomml.in/wp-content/uploads/2023/03/eff_banner.png)



## Steps
1. **Install Requirements**
```shell
pip install -r Requirements.txt
```
2. **Data Preprocessing**
```shell
python data_preprocessing.py
```

  * Read csv file of dataset 
  * filter dataset from minor classes (less than 900 image) to avoid data imbalance.
  * after filteration it become 11 class
  * add column for images directory
3. **Training** 
```shell
python training.py
```
  * `dataLoader.py` Building Custom DataLoader
  * Image size (112,112,3)
  * Data Augumentation (`RandomCrop`,`RandomHorizontalFlip`)
  * `build_models.py` for building models  
  * `train_eval.py` contain the main function for `Train` & `Eval`

  **Crateria of training**:
For choosing the best and proper model, I choose `Accuracy` as a metric of Evaluation.
Train each network with configurations:

  `Batch Size` = 16

  `Epochs` = 10

  `Input Image Size` = (112, 112, 3)

  `optimizer` SGD

4. Results


 \  | VGG16 | EfficientNet | DenseNet
--- | --- | --- | ---
Accuracy | **86.88%**| 84.28%|83.12

`VGG16` Get highest accuracy among models

