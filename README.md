# simple-dl
Deep Learning related code examples.

Dependence：
[Numpy](http://www.numpy.org/)
[Scipy](https://www.scipy.org/)
[Matplotlib](http://matplotlib.org/)
[Sklearn](http://scikit-learn.org/).


## Description 
   TIP: These are the UFLDL exercises implement in python.
   
   Move to [skyduy/DeepLearningTutorials](https://github.com/skyduy/DeepLearningTutorials) for more knowledge.
####SparseAutoencoder
 - Sparse autoencoder.
 - 稀疏自编码
   
####PCA_Whitening
 - PCA / Whitening based on PCA / ZCA Whitening based on PCA.
 - 主成分分析 / PCA白化 / ZCA白化
   
####SoftmaxRegression
 - SoftMax Regression, generalization of logistic regression.
 - SoftMax回归
   
####SelfTaughtLearning
 - Extract features by Autoencoder, then make SoftMax Regression.
 - 自编码  -->  SoftMax
   
####DeepNetworks
 - Extract features by two hidden layers Autoencoder (and Fine-tuning with the labeled data), then make SoftMax Regression.
 - 深度多层自编码 (--> 微调 ) --> SoftMax 
    
####LinearDecoder
 - ZCAWhiten the data and get the parameters of Autoencoder which has a linear decoder, then save the parameters. 
 - ZCA白化 --> 线性解码的稀疏自编码 --> 获取预处理参数
   
####ConvolutionPooling
 - Extract features with the parameters above and convolve/pool the features, then make SoftMax Regression.
 - 应用预处理参数 --> 卷积/池化 --> SoftMax
   


## Others

- Reference
    - [UFLDL](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)
   
- TODOS: 
    - Advanced Topics of UFLDL
        - Sparse Coding
        - ICA Style Models
    - ~~Other deep learning related code.~~: [skyduy/DeepLearningTutorials](https://github.com/skyduy/DeepLearningTutorials)
