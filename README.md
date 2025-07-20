## About

This repository contains a list of papers and my implementations of them in the field of AI (Mainly Language Modelling and Sequences Modelling) so that I can learn more about the corrent SoTA.

I will try to release an accompanying blog with each implement and provide my own explanations of each technique/architecture

The implementations will use PyTorch because it enables me to spend less time on setting up BackPropagation and get directly to replicating the paper.

The list of papers as of now is taken from [this repository](https://github.com/adam-maj/deep-learning) however, It will keep changing over time as new papers come along.

## Papers

**Deep Neural Networks**

- **DNN** - Learning Internal Representations by Error Propagation (1987), D. E. Rumelhart et al. [[PDF]](https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap8_PDP86.pdf)
- **CNN** - Backpropagation Applied to Handwritten Zip Code Recognition (1989), Y. Lecun et al. [[PDF]](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
- **LeNet** - Gradient-Based Learning Applied to Document Recognition (1998), Y. Lecun et al. [[PDF]](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- **AlexNet** - ImageNet Classification with Deep Convolutional Networks (2012), A. Krizhevsky et al. [[PDF]](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- **U-Net** - U-Net: Convolutional Networks for Biomedical Image Segmentation (2015), O. Ronneberger et al. [[PDF]](https://arxiv.org/abs/1505.04597)

### 2. Optimization and Regularization Techniques

#### Papers

- **Weight Decay** (1991): A Simple Weight Decay Can Improve Generalization [pdf](https://www.cs.toronto.edu/~hinton/absps/nips93.pdf)
- **ReLU** (2011): Deep Sparse Rectified Neural Networks [pdf](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)
- **Residuals** (2015): Deep Residual Learning for Image Recognition [pdf](https://arxiv.org/pdf/1512.03385.pdf)
- **Dropout** (2014): Preventing Neural Networks from Overfitting [pdf](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
- **BatchNorm** (2015): Accelerating Deep Network Training [pdf](https://arxiv.org/pdf/1502.03167.pdf)
- **LayerNorm** (2016): Layer Normalization [pdf](https://arxiv.org/pdf/1607.06450.pdf)
- **GELU** (2016): Gaussian Error Linear Units [pdf](https://arxiv.org/pdf/1606.08415.pdf)
- **Adam** (2014): Stochastic Optimization Method [pdf](https://arxiv.org/pdf/1412.6980.pdf)

### 3. Sequence Modeling

#### Papers

- **RNN** (1989): Continually Running Fully Recurrent Neural Networks [pdf](https://www.bioinf.jku.at/publications/older/2604.pdf)
- **LSTM** (1997): Long-Short Term Memory [pdf](https://www.bioinf.jku.at/publications/older/2308.pdf)
- **Learning to Forget** (2000): Continual Prediction with LSTM [pdf](https://www.researchgate.net/publication/221601044_Learning_to_Forget_Continual_Prediction_with_LSTM)
- **Word2Vec** (2013): Word Representations in Vector Space [pdf](https://arxiv.org/pdf/1301.3781.pdf)
- **Phrase2Vec** (2013): Distributed Representations of Words and Phrases [pdf](https://arxiv.org/pdf/1310.4546.pdf)
- **Encoder-Decoder** (2014): RNN Encoder-Decoder for Machine Translation [pdf](https://arxiv.org/pdf/1406.1078.pdf)
- **Seq2Seq** (2014): Sequence to Sequence Learning [pdf](https://arxiv.org/pdf/1409.3215.pdf)
- **Attention** (2014): Neural Machine Translation with Alignment [pdf](https://arxiv.org/pdf/1409.0473.pdf)
- **Mixture of Experts** (2017): Sparsely-Gated Neural Networks [pdf](https://arxiv.org/pdf/1701.06538.pdf)

### 4. Language Modeling

#### Papers

- **Transformer** (2017): Attention Is All You Need [pdf](https://arxiv.org/pdf/1706.03762.pdf)
- **BERT** (2018): Bidirectional Transformers for Language Understanding [pdf](https://arxiv.org/pdf/1810.04805.pdf)
- **RoBERTa** (2019): Robustly Optimized BERT Pretraining [pdf](https://arxiv.org/pdf/1907.11692.pdf)
- **T5** (2019): Unified Text-to-Text Transformer [pdf](https://arxiv.org/pdf/1910.10683.pdf)
- **GPT Series**:
  - GPT (2018): Generative Pre-Training [pdf](https://arxiv.org/pdf/1810.04805.pdf)
  - GPT-2 (2018): Unsupervised Multitask Learning [pdf](https://arxiv.org/pdf/1902.01082.pdf)
  - GPT-3 (2020): Few-Shot Learning [pdf](https://arxiv.org/pdf/2005.14165.pdf)
  - GPT-4 (2023): Advanced Language Model [pdf](https://arxiv.org/pdf/2303.08774.pdf)
- **LoRA** (2021): Low-Rank Adaptation of Large Language Models [pdf](https://arxiv.org/pdf/2106.09685.pdf)
- **RLHF** (2019): Fine-Tuning from Human Preferences [pdf](https://arxiv.org/pdf/1909.08593.pdf)
- **InstructGPT** (2022): Following Instructions with Human Feedback [pdf](https://arxiv.org/pdf/2203.02155.pdf)
- **Vision Transformer** (2020): Image Recognition with Transformers [pdf](https://arxiv.org/pdf/2010.11929.pdf)
- **ELECTRA** (2020): Discriminative Pre-training [pdf](https://arxiv.org/pdf/2003.10555.pdf)
