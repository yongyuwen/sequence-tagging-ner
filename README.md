# Keras-BiLSTMCRF

Keras Implementation of the BiLSTM-CRF model as described in https://guillaumegenthial.github.io/. 

For the PyTorch implementation (with ELMO) please refer to this [link](https://github.com/yongyuwen/sequence-tagging-ner).

## Usage
1.	**Requirements**:  
    a.	Packages: Anaconda, TensorFlow, Keras  
    b.	Data: Train, valid and test datasets in CoNLL 2003 NER format.  
    c.	Glove 300B embeddings (optional) 
    
2.	**Configure Settings**:  
    a.	Change settings in model/config.py  
    b.	Main settings to change: File directories, model hyperparameters etc.  
    
3.	**Build Data**:  
    a.	Run build_data.py  
        i.	Builds embedding dictionary, text file of words, chars tags, as well as idx to word and idx to char mapping for the model to read  
        
4.	**Train Model**:  
    a.	Run train_keras.py  
    
5.	**Test Model**:  
    a.	Run evaluate_keras.py  
    b.	Evaluates on test set. Also accepts other arguments to predict on custom string
