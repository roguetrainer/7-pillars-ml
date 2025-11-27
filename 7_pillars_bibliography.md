# The Seven Pillars of Statistical Wisdom in Modern AI/ML
## Comprehensive Bibliography & Resources

*A curated collection of foundational texts, seminal papers, models, courses, and interactive resources organized by pillar*

---

## Table of Contents

1. [Foundational Books](#foundational-books)
2. [Resources by Pillar](#resources-by-pillar)
3. [Models & Implementations](#models--implementations)
4. [Courses & Tutorials](#courses--tutorials)
5. [Interactive Resources](#interactive-resources)
6. [Research Groups & Labs](#research-groups--labs)
7. [Datasets](#datasets)

---

## Foundational Books

### Primary Source
- **Stigler, S. M.** (2016). *The Seven Pillars of Statistical Wisdom*. Harvard University Press.
  - ISBN: 978-0674088917
  - [Harvard University Press](https://www.hup.harvard.edu/catalog.php?isbn=9780674088917)
  - The original work identifying the seven conceptual pillars of statistics

### Statistical Foundations
- **Efron, B., & Hastie, T.** (2016). *Computer Age Statistical Inference: Algorithms, Evidence, and Data Science*. Cambridge University Press.
  - ISBN: 978-1108411851
  - [Free PDF from authors](https://web.stanford.edu/~hastie/CASI/)
  - Bridges classical statistics and modern computation

- **Casella, G., & Berger, R. L.** (2002). *Statistical Inference* (2nd ed.). Duxbury Press.
  - ISBN: 978-0534243128
  - Classic graduate-level text on statistical theory

- **Wasserman, L.** (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer.
  - ISBN: 978-0387402727
  - Compact introduction to modern statistics

### Machine Learning Theory
- **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press.
  - ISBN: 978-0262035613
  - [Free online version](https://www.deeplearningbook.org/)
  - Comprehensive deep learning textbook

- **Murphy, K. P.** (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
  - ISBN: 978-0262046824
  - [Free PDF](https://probml.github.io/pml-book/book1.html)
  - Modern probabilistic approach to ML

- **Murphy, K. P.** (2023). *Probabilistic Machine Learning: Advanced Topics*. MIT Press.
  - ISBN: 978-0262048439
  - [Free PDF](https://probml.github.io/pml-book/book2.html)
  - Advanced topics including deep learning

- **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
  - ISBN: 978-0387310732
  - Classic Bayesian approach to ML

- **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
  - ISBN: 978-0387848570
  - [Free PDF from authors](https://hastie.su.domains/ElemStatLearn/)
  - Classic statistical learning theory

### Practical Machine Learning
- **Géron, A.** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly.
  - ISBN: 978-1098125974
  - Practical guide with working code

- **Raschka, S., & Mirjalili, V.** (2019). *Python Machine Learning* (3rd ed.). Packt.
  - ISBN: 978-1789955750
  - Comprehensive practical guide

### Information Theory & Statistics
- **Cover, T. M., & Thomas, J. A.** (2006). *Elements of Information Theory* (2nd ed.). Wiley.
  - ISBN: 978-0471241959
  - Foundation for understanding likelihood and information

- **MacKay, D. J. C.** (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
  - ISBN: 978-0521642989
  - [Free PDF from author](http://www.inference.org.uk/mackay/itila/)
  - Connects information theory to ML

---

## Resources by Pillar

### Pillar 1: Aggregation

#### Key Papers
- **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
  - [DOI: 10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
  - Seminal paper introducing Random Forests

- **Breiman, L.** (1996). Bagging Predictors. *Machine Learning*, 24(2), 123-140.
  - [DOI: 10.1007/BF00058655](https://doi.org/10.1007/BF00058655)
  - Bootstrap Aggregating (Bagging)

- **Dietterich, T. G.** (2000). Ensemble Methods in Machine Learning. *MCS 2000: Multiple Classifier Systems*, 1-15.
  - [DOI: 10.1007/3-540-45014-9_1](https://doi.org/10.1007/3-540-45014-9_1)
  - Survey of ensemble methods

- **Shazeer, N., et al.** (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR 2017*.
  - [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
  - Mixture of Experts for neural networks

- **Fedus, W., et al.** (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*, 23(120), 1-39.
  - [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
  - Scaling MoE to trillion parameters

- **Jiang, A. Q., et al.** (2024). Mixtral of Experts. *arXiv preprint*.
  - [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
  - Modern open-source MoE architecture

#### Dropout Papers
- **Srivastava, N., et al.** (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(1), 1929-1958.
  - [JMLR Link](https://www.jmlr.org/papers/v15/srivastava14a.html)
  - Original dropout paper

- **Gal, Y., & Ghahramani, Z.** (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML 2016*.
  - [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)
  - Dropout as Bayesian inference

#### Models
- **Hugging Face:** `RandomForestClassifier` in scikit-learn
- **Hugging Face:** [`mistralai/Mixtral-8x7B-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- **Hugging Face:** [`mistralai/Mixtral-8x22B-v0.1`](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)

---

### Pillar 2: Information

#### Key Papers
- **Kaplan, J., et al.** (2020). Scaling Laws for Neural Language Models. *arXiv preprint*.
  - [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
  - OpenAI's original scaling laws paper

- **Hoffmann, J., et al.** (2022). Training Compute-Optimal Large Language Models. *arXiv preprint*.
  - [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
  - Chinchilla paper: optimal data/model size ratios

- **Hestness, J., et al.** (2017). Deep Learning Scaling is Predictable, Empirically. *arXiv preprint*.
  - [arXiv:1712.00409](https://arxiv.org/abs/1712.00409)
  - Early work on scaling laws

- **Henighan, T., et al.** (2020). Scaling Laws for Autoregressive Generative Modeling. *arXiv preprint*.
  - [arXiv:2010.14701](https://arxiv.org/abs/2010.14701)
  - Scaling laws for images, video, math

#### Data Efficiency
- **Brown, T., et al.** (2020). Language Models are Few-Shot Learners. *NeurIPS 2020*.
  - [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
  - GPT-3: Few-shot learning

- **Settles, B.** (2009). Active Learning Literature Survey. *Computer Sciences Technical Report 1648*, University of Wisconsin–Madison.
  - [PDF Link](http://burrsettles.com/pub/settles.activelearning.pdf)
  - Comprehensive active learning survey

- **Sorscher, B., et al.** (2022). Beyond neural scaling laws: beating power law scaling via data pruning. *NeurIPS 2022*.
  - [arXiv:2206.14486](https://arxiv.org/abs/2206.14486)
  - Data quality vs quantity

#### Information Theory Foundations
- **Shannon, C. E.** (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.
  - [DOI: 10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
  - Foundation of information theory

---

### Pillar 3: Likelihood

#### Key Papers
- **Bayes, T.** (1763). An Essay towards solving a Problem in the Doctrine of Chances. *Philosophical Transactions of the Royal Society of London*, 53, 370-418.
  - Historical foundation of Bayesian inference

- **Fisher, R. A.** (1922). On the Mathematical Foundations of Theoretical Statistics. *Philosophical Transactions of the Royal Society of London. Series A*, 222, 309-368.
  - Foundation of maximum likelihood estimation

- **Kingma, D. P., & Ba, J.** (2014). Adam: A Method for Stochastic Optimization. *ICLR 2015*.
  - [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
  - Adam optimizer (adaptive learning rates)

#### Contrastive Learning
- **Chen, T., et al.** (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*.
  - [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
  - SimCLR: contrastive learning for vision

- **He, K., et al.** (2020). Momentum Contrast for Unsupervised Visual Representation Learning. *CVPR 2020*.
  - [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
  - MoCo: momentum-based contrastive learning

- **Radford, A., et al.** (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.
  - [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
  - CLIP: contrastive language-image pretraining

- **Grill, J.-B., et al.** (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. *NeurIPS 2020*.
  - [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
  - BYOL: self-supervised without negative pairs

#### Energy-Based Models
- **LeCun, Y., et al.** (2006). A Tutorial on Energy-Based Learning. *Predicting Structured Data*, MIT Press.
  - [PDF Link](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
  - Foundation of energy-based models

- **Song, Y., & Ermon, S.** (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *NeurIPS 2019*.
  - [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
  - Score-based generative models

#### Models
- **Hugging Face:** [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
- **Hugging Face:** [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14)
- **Hugging Face:** [`facebook/moco-v3`](https://huggingface.co/facebook/moco-v3)

---

### Pillar 4: Intercomparison

#### Key Papers
- **Devlin, J., et al.** (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.
  - [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
  - Masked language modeling

- **He, K., et al.** (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*.
  - [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
  - MAE: masked autoencoding for images

- **Bao, H., et al.** (2021). BEiT: BERT Pre-Training of Image Transformers. *ICLR 2022*.
  - [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)
  - BERT-style pretraining for vision

#### Cross-Validation & Model Selection
- **Kohavi, R.** (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. *IJCAI 1995*.
  - [PDF Link](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf)
  - Classic cross-validation paper

- **Bergstra, J., & Bengio, Y.** (2012). Random Search for Hyper-Parameter Optimization. *JMLR*, 13, 281-305.
  - [JMLR Link](https://www.jmlr.org/papers/v13/bergstra12a.html)
  - Hyperparameter optimization

#### Self-Supervised Learning Surveys
- **Jing, L., & Tian, Y.** (2020). Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey. *IEEE TPAMI*, 43(11), 4037-4058.
  - [arXiv:1902.06162](https://arxiv.org/abs/1902.06162)
  - Comprehensive self-supervised learning survey

- **Liu, X., et al.** (2021). Self-supervised Learning: Generative or Contrastive. *IEEE TKDE*, 35(1), 857-876.
  - [arXiv:2006.08218](https://arxiv.org/abs/2006.08218)
  - Taxonomy of self-supervised methods

#### Models
- **Hugging Face:** [`bert-base-uncased`](https://huggingface.co/bert-base-uncased)
- **Hugging Face:** [`bert-large-uncased`](https://huggingface.co/bert-large-uncased)
- **Hugging Face:** [`roberta-base`](https://huggingface.co/roberta-base)
- **Hugging Face:** [`facebook/mae-base`](https://huggingface.co/facebook/mae-base)

---

### Pillar 5: Regression

#### Key Papers
- **Galton, F.** (1886). Regression Towards Mediocrity in Hereditary Stature. *Journal of the Anthropological Institute*, 15, 246-263.
  - Original "regression to the mean" paper

- **Tikhonov, A. N.** (1943). On the stability of inverse problems. *Doklady Akademii Nauk SSSR*, 39(5), 195-198.
  - Foundation of regularization (Tikhonov regularization)

- **Hoerl, A. E., & Kennard, R. W.** (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55-67.
  - [DOI: 10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
  - Ridge regression

- **Tibshirani, R.** (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
  - [DOI: 10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
  - L1 regularization (LASSO)

#### Regularization in Neural Networks
- **Krogh, A., & Hertz, J. A.** (1992). A Simple Weight Decay Can Improve Generalization. *NeurIPS 1991*.
  - Early weight decay paper

- **Nowlan, S. J., & Hinton, G. E.** (1992). Simplifying Neural Networks by Soft Weight-Sharing. *Neural Computation*, 4(4), 473-493.
  - [DOI: 10.1162/neco.1992.4.4.473](https://doi.org/10.1162/neco.1992.4.4.473)
  - Weight decay and regularization

- **Wan, L., et al.** (2013). Regularization of Neural Networks using DropConnect. *ICML 2013*.
  - [PDF Link](http://proceedings.mlr.press/v28/wan13.pdf)
  - DropConnect: drop connections not neurons

#### Batch Normalization
- **Ioffe, S., & Szegedy, C.** (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*.
  - [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
  - Batch normalization

- **Ba, J. L., Kiros, J. R., & Hinton, G. E.** (2016). Layer Normalization. *arXiv preprint*.
  - [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
  - Layer normalization

#### Early Stopping
- **Prechelt, L.** (1998). Early Stopping - But When? *Neural Networks: Tricks of the Trade*, Springer.
  - [DOI: 10.1007/3-540-49430-8_3](https://doi.org/10.1007/3-540-49430-8_3)
  - Practical guide to early stopping

---

### Pillar 6: Design

#### Key Papers
- **Fisher, R. A.** (1935). *The Design of Experiments*. Oliver and Boyd.
  - Foundation of experimental design and randomization

- **Robbins, H., & Monro, S.** (1951). A Stochastic Approximation Method. *The Annals of Mathematical Statistics*, 22(3), 400-407.
  - [DOI: 10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)
  - Foundation of stochastic optimization

- **Bottou, L.** (2010). Large-Scale Machine Learning with Stochastic Gradient Descent. *COMPSTAT 2010*.
  - [DOI: 10.1007/978-3-7908-2604-3_16](https://doi.org/10.1007/978-3-7908-2604-3_16)
  - Modern SGD practices

#### Curriculum Learning
- **Bengio, Y., et al.** (2009). Curriculum Learning. *ICML 2009*.
  - [DOI: 10.1145/1553374.1553380](https://doi.org/10.1145/1553374.1553380)
  - Original curriculum learning paper

- **Soviany, P., et al.** (2022). Curriculum Learning: A Survey. *International Journal of Computer Vision*, 130, 1526-1565.
  - [arXiv:2101.10382](https://arxiv.org/abs/2101.10382)
  - Comprehensive survey

#### Data Augmentation
- **Shorten, C., & Khoshgoftaar, T. M.** (2019). A Survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*, 6(1), 60.
  - [DOI: 10.1186/s40537-019-0197-0](https://doi.org/10.1186/s40537-019-0197-0)
  - Survey of augmentation techniques

- **Cubuk, E. D., et al.** (2019). AutoAugment: Learning Augmentation Strategies From Data. *CVPR 2019*.
  - [arXiv:1805.09501](https://arxiv.org/abs/1805.09501)
  - Learned augmentation policies

- **Cubuk, E. D., et al.** (2020). RandAugment: Practical Automated Data Augmentation with a Reduced Search Space. *CVPR Workshops 2020*.
  - [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
  - Simplified automated augmentation

#### Synthetic Data
- **Nikolenko, S. I.** (2021). Synthetic Data for Deep Learning. Springer.
  - [arXiv:1909.11512](https://arxiv.org/abs/1909.11512)
  - Comprehensive synthetic data overview

- **Gunasekar, S., et al.** (2023). Textbooks Are All You Need. *arXiv preprint*.
  - [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)
  - Phi-1: quality over quantity in pretraining

---

### Pillar 7: Residual

#### Key Papers
- **Friedman, J. H.** (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189-1232.
  - [DOI: 10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  - Gradient boosting foundation

- **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
  - [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
  - XGBoost algorithm

- **Ke, G., et al.** (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
  - [PDF Link](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
  - LightGBM

- **Prokhorenkova, L., et al.** (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 2018*.
  - [arXiv:1706.09516](https://arxiv.org/abs/1706.09516)
  - CatBoost

#### Residual Networks
- **He, K., et al.** (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
  - [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
  - Original ResNet paper - revolutionary architecture

- **He, K., et al.** (2016). Identity Mappings in Deep Residual Networks. *ECCV 2016*.
  - [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
  - ResNet v2 improvements

- **Huang, G., et al.** (2017). Densely Connected Convolutional Networks. *CVPR 2017*.
  - [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
  - DenseNet: connecting all layers

- **Zagoruyko, S., & Komodakis, N.** (2016). Wide Residual Networks. *BMVC 2016*.
  - [arXiv:1605.07146](https://arxiv.org/abs/1605.07146)
  - Wide ResNets

#### Transformers (Residual Connections Throughout)
- **Vaswani, A., et al.** (2017). Attention is All You Need. *NeurIPS 2017*.
  - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
  - Transformer architecture with residual connections

#### Error Analysis & Hard Negative Mining
- **Schroff, F., Kalenichenko, D., & Philbin, J.** (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. *CVPR 2015*.
  - [arXiv:1503.03832](https://arxiv.org/abs/1503.03832)
  - Hard negative mining in practice

- **Wu, C.-Y., et al.** (2017). Sampling Matters in Deep Embedding Learning. *ICCV 2017*.
  - [arXiv:1706.07567](https://arxiv.org/abs/1706.07567)
  - Strategic sampling of hard examples

#### Models
- **Hugging Face:** [`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50)
- **Hugging Face:** [`microsoft/resnet-152`](https://huggingface.co/microsoft/resnet-152)
- **XGBoost:** [Official GitHub](https://github.com/dmlc/xgboost)
- **LightGBM:** [Official GitHub](https://github.com/microsoft/LightGBM)
- **CatBoost:** [Official GitHub](https://github.com/catboost/catboost)

---

## Models & Implementations

### Hugging Face Model Collections

#### Language Models
- **BERT Family:**
  - [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) - 110M parameters
  - [`bert-large-uncased`](https://huggingface.co/bert-large-uncased) - 340M parameters
  - [`roberta-base`](https://huggingface.co/roberta-base) - Optimized BERT
  - [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) - Distilled BERT

- **GPT Family:**
  - [`gpt2`](https://huggingface.co/gpt2) - 117M parameters
  - [`gpt2-medium`](https://huggingface.co/gpt2-medium) - 345M parameters
  - [`gpt2-large`](https://huggingface.co/gpt2-large) - 774M parameters
  - [`gpt2-xl`](https://huggingface.co/gpt2-xl) - 1.5B parameters

- **Modern Open LLMs:**
  - [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf)
  - [`meta-llama/Llama-2-13b-hf`](https://huggingface.co/meta-llama/Llama-2-13b-hf)
  - [`mistralai/Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1)
  - [`mistralai/Mixtral-8x7B-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) - MoE
  - [`tiiuae/falcon-7b`](https://huggingface.co/tiiuae/falcon-7b)

#### Vision Models
- **ResNet Family:**
  - [`microsoft/resnet-18`](https://huggingface.co/microsoft/resnet-18)
  - [`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50)
  - [`microsoft/resnet-101`](https://huggingface.co/microsoft/resnet-101)
  - [`microsoft/resnet-152`](https://huggingface.co/microsoft/resnet-152)

- **Vision Transformers:**
  - [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224)
  - [`google/vit-large-patch16-224`](https://huggingface.co/google/vit-large-patch16-224)
  - [`facebook/mae-base`](https://huggingface.co/facebook/mae-base) - Masked Autoencoder

- **Multimodal:**
  - [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
  - [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14)

### GitHub Repositories

#### Classic ML Libraries
- **Scikit-learn:** [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
  - Random Forests, Ridge Regression, Cross-validation

- **XGBoost:** [https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)
  - Gradient boosting implementation

- **LightGBM:** [https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
  - Fast gradient boosting

- **CatBoost:** [https://github.com/catboost/catboost](https://github.com/catboost/catboost)
  - Gradient boosting with categorical features

#### Deep Learning Frameworks
- **PyTorch:** [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
  - Primary framework for research

- **PyTorch Examples:** [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
  - Official example implementations

- **TensorFlow:** [https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
  - Google's ML framework

- **Transformers (Hugging Face):** [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
  - State-of-the-art NLP models

#### Paper Implementations
- **Papers with Code:** [https://paperswithcode.com/](https://paperswithcode.com/)
  - Links papers to official implementations

- **Annotated Transformer:** [https://nlp.seas.harvard.edu/2018/04/03/attention.html](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
  - Line-by-line annotated transformer implementation

- **The Annotated Diffusion Model:** [https://huggingface.co/blog/annotated-diffusion](https://huggingface.co/blog/annotated-diffusion)
  - Annotated DDPM implementation

---

## Courses & Tutorials

### University Courses (Free Online)

#### Stanford
- **CS229: Machine Learning** - Andrew Ng
  - [Course Website](http://cs229.stanford.edu/)
  - [YouTube Lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
  - Classical ML foundations

- **CS231n: Convolutional Neural Networks for Visual Recognition**
  - [Course Website](http://cs231n.stanford.edu/)
  - [YouTube Lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
  - Deep learning for computer vision

- **CS224n: Natural Language Processing with Deep Learning**
  - [Course Website](https://web.stanford.edu/class/cs224n/)
  - [YouTube Lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)
  - Modern NLP with transformers

- **CS330: Deep Multi-Task and Meta Learning**
  - [Course Website](https://cs330.stanford.edu/)
  - Meta-learning and transfer learning

#### MIT
- **6.S191: Introduction to Deep Learning**
  - [Course Website](http://introtodeeplearning.com/)
  - [YouTube Lectures](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
  - Fast-paced intro to deep learning

- **6.036: Introduction to Machine Learning**
  - [OCW Course](https://ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020/)
  - Foundations of ML

#### Berkeley
- **CS182/282A: Designing, Visualizing and Understanding Deep Neural Networks**
  - [Course Website](https://cs182sp21.github.io/)
  - Deep learning theory and practice

- **CS188: Introduction to Artificial Intelligence**
  - [Course Website](https://inst.eecs.berkeley.edu/~cs188/)
  - Broad AI foundations

#### Carnegie Mellon
- **10-301/601: Introduction to Machine Learning**
  - [Course Website](https://www.cs.cmu.edu/~10601/)
  - Core ML course

- **11-785: Introduction to Deep Learning**
  - [Course Website](https://deeplearning.cs.cmu.edu/)
  - Comprehensive deep learning

### Online Learning Platforms

#### Fast.ai
- **Practical Deep Learning for Coders**
  - [Course Website](https://course.fast.ai/)
  - [Free Book](https://github.com/fastai/fastbook)
  - Top-down practical approach

- **Deep Learning from the Foundations**
  - [Course Website](https://course.fast.ai/part2)
  - Bottom-up implementation

#### Coursera
- **Deep Learning Specialization** - Andrew Ng
  - [Course Link](https://www.coursera.org/specializations/deep-learning)
  - 5-course sequence covering foundations

- **Machine Learning Specialization** - Andrew Ng
  - [Course Link](https://www.coursera.org/specializations/machine-learning-introduction)
  - Updated ML foundations

#### deeplearning.ai
- [https://www.deeplearning.ai/](https://www.deeplearning.ai/)
- Multiple specializations on various topics

### Specialized Topics

#### Reinforcement Learning
- **Deep RL Course** - Hugging Face
  - [Course Website](https://huggingface.co/deep-rl-course)
  - Hands-on RL with modern tools

- **CS285: Deep Reinforcement Learning** - UC Berkeley
  - [Course Website](http://rail.eecs.berkeley.edu/deeprlcourse/)
  - Graduate-level RL

#### Transformers & LLMs
- **Hugging Face NLP Course**
  - [Course Website](https://huggingface.co/learn/nlp-course)
  - Free course on transformers

- **LLM University** - Cohere
  - [Course Website](https://docs.cohere.com/docs/llmu)
  - LLM fundamentals

---

## Interactive Resources

### Visualization & Explanation Sites

#### Distill.pub
- [https://distill.pub/](https://distill.pub/)
- High-quality interactive ML explanations
- **Key Articles:**
  - [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
  - [Feature Visualization](https://distill.pub/2017/feature-visualization/)
  - [Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/)
  - [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/)
  - [Exploring Neural Networks with Activation Atlases](https://distill.pub/2019/activation-atlas/)

#### Jay Alammar's Blog
- [https://jalammar.github.io/](https://jalammar.github.io/)
- Visual explanations of transformers
- **Key Posts:**
  - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
  - [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
  - [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

#### 3Blue1Brown
- [Neural Networks YouTube Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Beautiful mathematical visualizations
- Excellent intuition building

#### TensorFlow Playground
- [https://playground.tensorflow.org/](https://playground.tensorflow.org/)
- Interactive neural network visualization
- Experiment with architectures in browser

#### Seeing Theory
- [https://seeing-theory.brown.edu/](https://seeing-theory.brown.edu/)
- Visual introduction to probability and statistics
- Beautiful interactive visualizations

#### CNN Explainer
- [https://poloclub.github.io/cnn-explainer/](https://poloclub.github.io/cnn-explainer/)
- Interactive visualization of CNNs
- Great for understanding convolutions

#### The Neural Network Zoo
- [https://www.asimovinstitute.org/neural-network-zoo/](https://www.asimovinstitute.org/neural-network-zoo/)
- Visual catalog of neural network architectures

### Interactive Notebooks

#### Google Colab Notebooks
- [Colab Research](https://colab.research.google.com/)
- Free GPU/TPU access for experimentation

#### Kaggle
- [https://www.kaggle.com/](https://www.kaggle.com/)
- Datasets, competitions, and notebooks
- Learn from winning solutions

#### Papers with Code
- [https://paperswithcode.com/](https://paperswithcode.com/)
- Papers linked to code implementations
- Browse by task and dataset

---

## Research Groups & Labs

### Academic Research Groups

#### Stanford
- **Stanford AI Lab (SAIL)**
  - [https://ai.stanford.edu/](https://ai.stanford.edu/)
- **Stanford NLP Group**
  - [https://nlp.stanford.edu/](https://nlp.stanford.edu/)

#### MIT
- **MIT CSAIL**
  - [https://www.csail.mit.edu/](https://www.csail.mit.edu/)
- **MIT-IBM Watson AI Lab**
  - [https://mitibmwatsonailab.mit.edu/](https://mitibmwatsonailab.mit.edu/)

#### Berkeley
- **Berkeley AI Research (BAIR)**
  - [https://bair.berkeley.edu/](https://bair.berkeley.edu/)

#### CMU
- **CMU Machine Learning Department**
  - [https://www.ml.cmu.edu/](https://www.ml.cmu.edu/)

#### Toronto
- **Vector Institute**
  - [https://vectorinstitute.ai/](https://vectorinstitute.ai/)

#### Oxford
- **Oxford Machine Learning Research Group**
  - [https://www.robots.ox.ac.uk/~parg/](https://www.robots.ox.ac.uk/~parg/)

### Industry Research Labs

#### OpenAI
- [https://openai.com/research](https://openai.com/research)
- GPT series, CLIP, DALL-E

#### Google DeepMind
- [https://www.deepmind.com/research](https://www.deepmind.com/research)
- AlphaGo, AlphaFold, Transformers

#### Meta AI (FAIR)
- [https://ai.meta.com/research/](https://ai.meta.com/research/)
- PyTorch, BERT variants, LLaMA

#### Microsoft Research
- [https://www.microsoft.com/en-us/research/](https://www.microsoft.com/en-us/research/)
- ResNet, Turing-NLG

#### Anthropic
- [https://www.anthropic.com/research](https://www.anthropic.com/research)
- Claude, AI safety research

---

## Datasets

### Standard Benchmarks

#### Computer Vision
- **ImageNet**
  - [https://www.image-net.org/](https://www.image-net.org/)
  - 1.4M images, 1000 classes

- **CIFAR-10/100**
  - [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
  - Small-scale image classification

- **MNIST**
  - [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
  - Handwritten digits (classic)

- **COCO (Common Objects in Context)**
  - [https://cocodataset.org/](https://cocodataset.org/)
  - Object detection, segmentation, captioning

#### Natural Language Processing
- **GLUE Benchmark**
  - [https://gluebenchmark.com/](https://gluebenchmark.com/)
  - General language understanding tasks

- **SuperGLUE**
  - [https://super.gluebenchmark.com/](https://super.gluebenchmark.com/)
  - More challenging language tasks

- **SQuAD (Stanford Question Answering)**
  - [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
  - Reading comprehension

- **The Pile**
  - [https://pile.eleuther.ai/](https://pile.eleuther.ai/)
  - 825 GiB diverse text for language modeling

#### Multimodal
- **Conceptual Captions**
  - [https://ai.google.com/research/ConceptualCaptions/](https://ai.google.com/research/ConceptualCaptions/)
  - Image-text pairs

- **LAION-5B**
  - [https://laion.ai/blog/laion-5b/](https://laion.ai/blog/laion-5b/)
  - 5.85B image-text pairs

### Dataset Repositories
- **Hugging Face Datasets:** [https://huggingface.co/datasets](https://huggingface.co/datasets)
- **Kaggle Datasets:** [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
- **Papers with Code Datasets:** [https://paperswithcode.com/datasets](https://paperswithcode.com/datasets)
- **Google Dataset Search:** [https://datasetsearch.research.google.com/](https://datasetsearch.research.google.com/)

---

## Additional Resources

### Blogs & News

#### ML Research Blogs
- **OpenAI Blog:** [https://openai.com/blog](https://openai.com/blog)
- **Google AI Blog:** [https://ai.googleblog.com/](https://ai.googleblog.com/)
- **Meta AI Blog:** [https://ai.meta.com/blog/](https://ai.meta.com/blog/)
- **DeepMind Blog:** [https://www.deepmind.com/blog](https://www.deepmind.com/blog)
- **Anthropic Research:** [https://www.anthropic.com/research](https://www.anthropic.com/research)

#### Individual Researcher Blogs
- **Andrej Karpathy:** [https://karpathy.github.io/](https://karpathy.github.io/)
- **Christopher Olah:** [https://colah.github.io/](https://colah.github.io/)
- **Sebastian Ruder:** [https://www.ruder.io/](https://www.ruder.io/)
- **Lil'Log (Lilian Weng):** [https://lilianweng.github.io/](https://lilianweng.github.io/)

#### News Aggregators
- **Papers with Code:** [https://paperswithcode.com/](https://paperswithcode.com/)
- **arXiv Sanity:** [http://www.arxiv-sanity.com/](http://www.arxiv-sanity.com/)
- **Hugging Face Daily Papers:** [https://huggingface.co/papers](https://huggingface.co/papers)

### Podcasts
- **The TWIML AI Podcast:** [https://twimlai.com/](https://twimlai.com/)
- **Lex Fridman Podcast:** [https://lexfridman.com/podcast/](https://lexfridman.com/podcast/)
- **The Robot Brains Podcast:** [https://www.therobotbrains.ai/](https://www.therobotbrains.ai/)

### Communities
- **r/MachineLearning:** [https://www.reddit.com/r/MachineLearning/](https://www.reddit.com/r/MachineLearning/)
- **Hugging Face Forums:** [https://discuss.huggingface.co/](https://discuss.huggingface.co/)
- **MLOps Community:** [https://mlops.community/](https://mlops.community/)
- **Papers with Code Community:** [https://paperswithcode.com/](https://paperswithcode.com/)

---

## Keeping Current

### Preprint Servers
- **arXiv.org:** [https://arxiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent)
  - CS.LG (Machine Learning)
  - CS.AI (Artificial Intelligence)
  - CS.CV (Computer Vision)
  - CS.CL (Computation and Language)

### Conference Proceedings
- **NeurIPS:** [https://neurips.cc/](https://neurips.cc/)
- **ICML:** [https://icml.cc/](https://icml.cc/)
- **ICLR:** [https://iclr.cc/](https://iclr.cc/)
- **CVPR:** [https://cvpr.thecvf.com/](https://cvpr.thecvf.com/)
- **ACL:** [https://www.aclweb.org/](https://www.aclweb.org/)
- **EMNLP:** [https://2023.emnlp.org/](https://2023.emnlp.org/)

### Twitter/X Lists (AI/ML Researchers)
Follow lists curated by the community for latest research discussions

---

## Citation

If you use this bibliography or the Seven Pillars project in your work, please cite:

```bibtex
@misc{seven_pillars_ml_2024,
  title={The Seven Pillars of Statistical Wisdom in Modern AI/ML: A Comprehensive Guide},
  author={Your Name},
  year={2024},
  howpublished={GitHub/Personal Website}
}
```

---

## License & Contributing

This bibliography is an open educational resource. Contributions, corrections, and suggestions are welcome.

**Maintained by:** [Your Name]  
**Last Updated:** November 2024  
**Version:** 1.0

---

*"Standing on the shoulders of statistical giants."*
