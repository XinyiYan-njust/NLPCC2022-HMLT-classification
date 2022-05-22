# NLPCC2022 HMLT-classification
 Multi-label Classification Model for English Scientific Literature: develop a multi-label classification model for scientific research literature based on the given metadata (title and abstract) of scientific research literature and corresponding hierarchical labels from a domain-specific topic taxonomy.
 
 ## Code

The code for these Multi-label Classification experiments are adapted from the [Named Entity Recognition Tool repository](https://github.com/glample/tagger).
We have improved the input data format of the code, extending the multi-level labels from the numbers to the text, and encoding the text category labels.

This hierarchical multi-label text classification model code is quoted from: Hierarchical Multi-Label Text Classification项目(链接：https://github.com/RandolphVI/Hierarchical-Multi-Label-Text-Classification)
The paper link corresponding to the code is: (https://dl.acm.org/citation.cfm?id=3357384.3357885). 


## Requirements

- Python 3
- Tensorflow 1.15.0
- Tensorboard 1.15.0
- Sklearn 0.19.1
- Numpy 1.16.2
- Gensim 3.8.3
- Tqdm 4.49.0


## Project

项目的架构如下:

```项目根目录
.
├── HARNN
│   ├── train_harnn.py
│   ├── text_harnn.py
│   ├── test_harnn.py
│   └── visualization.py
├── utils
│   ├── checkmate.py
│   ├── param_parser.py
│   └── data_helpers.py
├── data
│   ├── index_to_labels
│   │    ├── label_index
│   │    │    ├── level1_dict.json
│   │    │    ├── level2_dict.json
│   │    │    └── level3_dict.json
│   │    ├── index_to_labels.py
│   │    └── verification set.json
│   ├── evaluation code
│   │    ├── data
│   │    │    ├── labels_1.rand123
│   │    │    ├── labels_2.rand123
│   │    │    ├── labels_3.rand123
│   │    │    └── labels_all.rand123
│   │    ├── input_sample
│   │    └── eval.py
│   ├── word2vec_100.kv
│   ├── Test_sample.json
│   ├── Train_sample.json
│   └── Validation_sample.json
├── README.md
└── requirements.txt
```

## model

Since the existing multi-label classification models ignore the correlation between the labels, moreover, the existing multi-label classification models do not take into account that different parts of the text have different contributions to different labels when predicting the labels.
For the above two motivations, the corresponding strategies are adopted to solve them:
1）Using RNN is decoding to generate the labels of the input text.This transforms the classification task to a seq2seq task.When decoding, the subsequent label generation needs to rely on the previously predicted labels, and the correlation between the labels is reflected here;
2）The attention mechanism was added to the middle of the seq2seq codec, making the generation of each tag depends dependent on different parts of the input sequence.

Before defining the HMTC problem, we start with a detailed description of the hierarchy and documentation.
defination 1：Hierarchical Structure。
    Given the possible class C, C is the H-level C= (C2, C2, …, CH), where C is the categories label set, H is the number of levels of label, and C^i is the set of i-layer label.
defination 2：HMTC Problem
    Given the document set D, and the associated hierarchical label structure \gamma, the HMTC problem can be turned to learn a classification model \Omega for label prediction, namely:
    ![](https://math.jianshu.com/math?formula=%5COmega(D%2C%5Cgamma%2C%5CTheta)%20%5Cto%20L)
    Where \ theta is the parameter to learn, Di is the i-th text, composed of N sequence words; corresponding to Li, li is the set of i-layer labels.
    In fact, there are some limitations to solving the HTMC task scenario: corresponding to the input text x, it is labeled in the H layer label system, and the number of labels in each layer is 1 or more.

The overall model architecture diagram is as follows:
    ![](https://upload-images.jianshu.io/upload_images/20501870-67d2810e0086f4df.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

  It is divided into three following layers:
  (1)Documentation Representing Layer (DRL) ———— Study the representation of text and hierarchical labels;
  (2)Hierarchical Attention-based Recurrent Layer (HARL) ———— Use the attention mechanism, let the learned text vector and label vector for loop learning, interaction;
  (3)Hybrid Predicting Layer (HPL) ———— Label prediction is performed in a mixed fashion.
  
  Here are these three parts:
  
  (1) Documentation Representing Layer
      On the text representation, the word vector is obtained using worde2vec, then the Bi-LSTM network for representation learning to obtain the sequence vector V.
      ![](https://upload-images.jianshu.io/upload_images/20501870-46e60ee013172dae.png?imageMogr2/auto-orient/strip|imageView2/2/w/280/format/webp)
      For subsequent operations, a word-based average pooling operation (word-wise average pooling operation) is used to turn V into a \tilde {V}.
      ![](https://upload-images.jianshu.io/upload_images/20501870-f9cc0836180dda90.png?imageMogr2/auto-orient/strip|imageView2/2/w/279/format/webp)
      On the hierarchical label characterization, lookup is used to generate the initialization matrix standard S
      ![](https://upload-images.jianshu.io/upload_images/20501870-ac1e5d3599f1750f.png?imageMogr2/auto-orient/strip|imageView2/2/w/182/format/webp)
      Finally, the representational learned V and S are spliced to the next layer of learning.
  
  (2) Hierarchical Attention-based Recurrent Layer
      This layer is the embodiment of the core of the model, and the main idea is to connect the first layer learning vector to a rnn network, which is inside the HAM structure, and HAM is called Hierarchical Attention-based Memory in the text, which means a hierarchical memory unit based on attention.In addition, the number of nodes in this circular network should be the hierarchy of labels. For example, if the label of the data set has 5 levels, then the rnn node in this layer is 5, which can be understood as progressive learning layer after layer, like the hierarchical structure of labels.
      The following figure is the HAM schematic diagram, somewhat similar to the LSTM structure, with three parts: Text-Category Attention (TCA), Class Prediction Module (CPM), Class Dependency Module (CDM):
      ![](https://upload-images.jianshu.io/upload_images/20501870-a41873ed69dd3aa4.png?imageMogr2/auto-orient/strip|imageView2/2/w/565/format/webp)
      The calculation formula is follows:
      ![](https://upload-images.jianshu.io/upload_images/20501870-2a0e017353551082.png?imageMogr2/auto-orient/strip|imageView2/2/w/502/format/webp)
      Where r_ {aat} ^ {h}, W_ {att} ^ h represents the text-label interaction information of the h-level layer, and the attention weight for the interaction with the h-level layer text and label, respectively.
      P_L ^ h, A_L ^ h respectively represents the probability of h-level layer label prediction, with information on the overall representation of the h-leve layer model.
      \ omega ^ h transmits the information learned by the h-layer as memory information.
      <1> Text-Category Attention
      The following figure is a Text-Category Attention computational graph, whose main goal is to learn the input text interactively with the labels at all levels, using a method similar to the attention mechanism.
      ![](https://upload-images.jianshu.io/upload_images/20501870-93767c9f5eb32d35.png?imageMogr2/auto-orient/strip|imageView2/2/w/595/format/webp)
      The calculation formula corresponds as follows: V_h is updated with the information of the previous layer to accept the information related to the label; O_h, W_ {att} ^ h is the attention of the input text and the h layer label as the weight value; M_h, r_ {att} ^ h calculates the labeled text information and obtains the final text representation information.
      ![](https://upload-images.jianshu.io/upload_images/20501870-861faba1df222591.png?imageMogr2/auto-orient/strip|imageView2/2/w/202/format/webp)
      <2> Class Prediction Module
      The purpose of this module is to combine the original text semantic representation with the associated text category representation of the previous layer of information to generate the overall representation and predict the category of each layer, with the update formula:
      ![](https://upload-images.jianshu.io/upload_images/20501870-cc71f15b917081f3.png?imageMogr2/auto-orient/strip|imageView2/2/w/452/format/webp)
      <3> Class Dependency Module
      The purpose of this module is to interactive transfer learning the dependencies between different levels by retaining the hierarchical information of all levels. It mainly means to learn the degree of the association of each sequence in the text to the label in all levels, and to learn the information repeatedly.
      ![](https://upload-images.jianshu.io/upload_images/20501870-8bb9061d221d46d4.png?imageMogr2/auto-orient/strip|imageView2/2/w/407/format/webp)
 
  (3) Hybrid Predicting Layer
      Use the information of the second layer for the hybrid prediction. Why is it a hybrid prediction?The reason is: In the author's opinion, the previous learned each layer P_L ^ h prediction is only a local prediction, and each information needs to be combined to make a global prediction P_G:
      ![](https://upload-images.jianshu.io/upload_images/20501870-88f3d6fdec0715a0.png?imageMogr2/auto-orient/strip|imageView2/2/w/497/format/webp)
      The local and global predicted values are then weighted as the final predicted value P_F:
      ![](https://upload-images.jianshu.io/upload_images/20501870-3ba08bf14a587ef1.png?imageMogr2/auto-orient/strip|imageView2/2/w/526/format/webp)
      Here the threshold \ alpha is taken at 0.5, considered locally as important as global.
  
  (4) Loss Function
      When local and global predictions are used, the authors make two loss function accordingly, the first is the loss of label prediction at each level, the second is the loss of global label prediction, and the last two add up and add a L2 regular as the final loss.
      ![](https://upload-images.jianshu.io/upload_images/20501870-3c43b1e6a1a9c025.png?imageMogr2/auto-orient/strip|imageView2/2/w/460/format/webp)


## Experiment

First, the trial data should be preprocessed to accommodate the model requirements.

（1）English text segmentation.Participle the title and summary using the nltk package and write the participle result to the source file.

（2）Text category label encoding processing.Turning the labels under the level list into digital format, the number of primary labels (level1) is 21, the number of secondary labels (level2) is 298, the number of tertiary labels (level3) is 1475, and the total number of labels (levels) types is 1794.Unlike the number of labels given in the document, the published trial dataset "Table of label dependencies.xlsx" will prevail.
        Therefore, when corresponding the number of class labels in level1 to one-to-one between 1-21, encoding 0 indicates no such category to accommodate the case where some categories have only secondary labels and no tertiary labels, such as:
        {"Physical chemistry": 1, "Inorganic chemistry": 2, "Cross-disciplinary concepts": 3}。
        Similarly, level2 and level3 do the same treatment.For levels, because it is the summary of the above three categories of labels, all the labels need to be processed to some extent. Its coding mode is: the first-level label coding is unchanged,
Direct writing; the secondary label coding is the original value plus the number of primary labels before writing; the tertiary label coding is converted to the original value plus the number of primary labels and secondary label bibliography before writing.in compliance with:
		{"id": "1", "title": ["Electromagnetic", "Confinement"], "abstract": ["We", "investigate", "electromagnetic"], "level1": [1, 3, 10], "level2": [6, 14, 48], "level3": [215, 216, 544], "levels": [1, 3, 10, 27, 35, 69, 534, 535, 863]}
		Eventually all the data is saved as a dictionary list in the json file.

（3）Adjust the participation to determine the results.The best outcome was determined by reference tuning.

（4）Decoding and model evaluation.Based on the result "predictions.json" from the model training, the result format is as follows:
    {"id": "1", "labels": [1, 3, 10, 27, 35, 69, 534, 535, 863, 1795], "predict_labels": [1, 3, 4, 10, 70, 1795, 1797], "predict_scores": [0.9549, 0.6582, 0.6729, 0.5173, 0.3687, 0.5, 0.4999]}；
    It is decoded and corresponding to the original data to obtain:
    {"title": "Electromagnetic Confinement via Spin–Orbit Interaction in Anisotropic Dielectrics", "abstract": "We investigate electromagnetic propagation in uniaxial dielectrics with a transversely varying orientation of the optic axis, the latter staying orthogonal everywhere in the propagation direction. In such a geometry, the field experiences no refractive index gradients, yet it acquires a transversely modulated Pancharatnam–Berry phase, that is, a geometric phase originating from a spin–orbit interaction. We show that the periodic evolution of the geometric phase versus propagation gives rise to a longitudinally invariant effective potential. In certain configurations, this geometric phase can provide transverse confinement and waveguiding. The theoretical findings are tested and validated against numerical simulations of the complete Maxwell’s equations. Our results introduce and illustrate the role of geometric phases on electromagnetic propagation over distances well exceeding the diffraction length, paving the way to a whole new family of guided waves and waveguides that do not rely on refractive index tailoring.", "level1": ["Physical chemistry", "Theoretical and computational chemistry", "Cross-disciplinary concepts"], "level2": ["Physical and chemical properties", "Optics", "Quantum mechanics"], "level3": ["Photonics", "Optical properties", "Polarization"], "levels": ["Physical chemistry", "Theoretical and computational chemistry", "Cross-disciplinary concepts", "Physical and chemical properties", "Optics", "Quantum mechanics", "Photonics", "Optical properties", "Polarization"], "pred_labels": ["Physical chemistry", "Cross-disciplinary concepts", "Materials science", "Theoretical and computational chemistry", "Physical and chemical processes"]}
    再使用eval.py进行模型的评估。

## Usage

 ### Input and output options

 ```
  --train-file              STR    Training file.      		Default is `../data/Train_sample.json`.
  --validation-file         STR    Validation file.      	Default is `../data/Validation_sample.json`.
  --test-file               STR    Testing file.       		Default is `../data/Test_sample.json`.
  --word2vec-file           STR    Word2vec model file.		Default is `../data/word2vec_100.kv`.
 ```

 ### Model option

 ```
  --pad-seq-len             INT     Padding Sequence length of data.                    Depends on data.
  --embedding-type          INT     The embedding type.                                 Default is 1.
  --embedding-dim           INT     Dim of character embedding.                         Default is 100.
  --lstm-dim                INT     Dim of LSTM neurons.                                Default is 256.
  --lstm-layers             INT     Number of LSTM layers.                              Defatul is 1.
  --attention-dim           INT     Dim of Attention neurons.                           Default is 200.
  --attention-penalization  BOOL    Use attention penalization or not.                  Default is True.
  --fc-dim                  INT     Dim of FC neurons.                                  Default is 512.
  --dropout-rate            FLOAT   Dropout keep probability.                           Default is 0.5.
  --alpha                   FLOAT   Weight of global part in loss cal.                  Default is 0.5.
  --num-classes-list        LIST    Each number of labels in hierarchical structure.    Depends on data.
  --total-classes           INT     Total number of labels.                             Depends on data.
  --topK                    INT     Number of top K prediction classes.                 Default is 5.
  --threshold               FLOAT   Threshold for prediction classes.                   Default is 0.5.
 ```

 ### Training option

 ```
  --epochs                  INT     Number of epochs.                       Default is 20.
  --batch-size              INT     Batch size.                             Default is 256.
  --learning-rate           FLOAT   Adam learning rate.                     Default is 0.001.
  --decay-rate              FLOAT   Rate of decay for learning rate.        Default is 0.95.
  --decay-steps             INT     How many steps before decy lr.          Default is 500.
  --evaluate-steps          INT     How many steps to evluate val set.      Default is 50.
  --l2-lambda               FLOAT   L2 regularization lambda.               Default is 0.0.
  --checkpoint-steps        INT     How many steps to save model.           Default is 50.
  --num-checkpoints         INT     Number of checkpoints to store.         Default is 10.
 ```

 ### Training
The commands for training the model are as follows:
 ```bash
 $ python3 train_harnn.py
 ```
Train the model with 20 epochs s and set the batch size to 256:
 ```bash
 $ python3 train_harnn.py --epochs 20 --batch-size 256
 ```
 ** Then you need to choose to train or store the model（T/R）**
 After training, you get the ` / log ` and ` / run ` folders.
 - `/log` The Folder is used to save the log information for the model.
 - `/run` The Folder is used to save the breakpoints.
 
 ### Test
The commands to test the model are as follows:
 ```bash
 $ python3 test_harnn.py
 ```
 Then you need to enter the name of the model (a 10-digit code, such as: 1652786880):
 Then you need to choose whether the best training model or the nearest training model (B for Best, L for Latest):
 Eventually you get the `predictions.json` file under the ` / outputs` folder.

 ### Evaluation
 Convert the predicted digital tag into text form first (see at index_to_labels/index_to_labels.py)
 The structure predicted by the model was again evaluated with the eval.py
 The evaluation results of commissioning are provided as follows:HMLT classification/data/evaluation code/input_sample/1652786880/RESULT.txt

## Reference
```bibtex
@inproceedings{huang2019hierarchical,
  author    = {Wei Huang and
               Enhong Chen and
               Qi Liu and
               Yuying Chen and
               Zai Huang and
               Yang Liu and
               Zhou Zhao and
               Dan Zhang and
               Shijin Wang},
  title     = {Hierarchical Multi-label Text Classification: An Attention-based Recurrent Network Approach},
  booktitle = {Proceedings of the 28th {ACM} {CIKM} International Conference on Information and Knowledge Management, {CIKM} 2019, Beijing, CHINA, Nov 3-7, 2019},
  pages     = {1051--1060},
  year      = {2019},
}```

