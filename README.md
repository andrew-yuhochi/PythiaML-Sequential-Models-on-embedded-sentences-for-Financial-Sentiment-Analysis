# PythiaML
PythiaML is a project scheme aiming at constructing a dynamic portfolio using Machine learning. The sophisticated final models are expected to include features from various sources, including but not limited to: market and technical indicators, company-wide financial ratios, regional economic statistics, and investor’s sentiment toward markets.


# Directory:
1. Basic Natural Language Processing on News Title to classify Stock Trend (https://github.com/andrew-yuhochi/Basic-Natural-Language-Processing-on-News-Title-to-classify-Stock-Trend
2. First attempt: Forecast the stock signal by NLP-based features (https://github.com/andrew-yuhochi/First-attempt-Forecast-the-stock-signal-by-NLP-based-features)
3. Sequential Models on embedded sentences for Financial Sentiment Analysis (you are here!)


## 3. Sequential Models on embedded sentences for Financial Sentiment Analysis

#### Date: 
28 Mar 2021 (HKT/ GMT +8)

#### Team: YU, Ho Chi Andrew. NG, Kwok Ching Arnold. ZHANG, Ruiqi Rachel. LIN, Haoli Horry.


#### Abstract
Financial Sentiment Analysis (FSA) aims to classify a sentence of financial text as expressing positive or negative opinions toward certain instance. It could be utilized for extracting investor and public expectation towards the market or a particular instrument via news reporting and social media. However, FSA is proved to be much more difficult than classifying general sentences such as tweets or reviews which have been successfully solved in recent years. The bottleneck of FSA is mainly attributed to the lack of reliable and high-quality labelled data and thus, transfer learning with domain adoption is nearly a must. The project starts by employing a pre-trained word embedding model to convert all input sentence to a matrix. Several sequential architectures of neural networks were then experimented to general (tweets and product reviews) data and formed base models for sentiment analysis. Next, the weightings and architectures of base models were inherited to domain (financial text) data for fine-tuning and domain adoption. Two transfer learning method “Fine Tuning the final layer” and “Transfer as initial” were experimented separately and showed the successes in FSA while also reflecting potential areas of insufficient domain adoption for further consideration and improvision.

#### Dataset
Four datasets were used in this project and all of them are well known benchmark datasets in the fields of FSA or general sentiment analysis. “Sentiment140” and “Amazon Reviews Polarity” are used for training the base models. Both datasets consist of a text column that were used as the input in this project and the corresponding label of either 1 or 0 representing a positive or negative sentiment respectively. Combining these two datasets, we have almost 5.6 millions of balanced data in total. In the later section, we will call this combined dataset as “general data”. 

“Financial PhraseBank” and “FiQA Task 1” are then used for fine tuning the base models in the later section. As same as before, these datasets consist of a text column that was used as the input and a label representing the corresponding sentiment. However, the label of “Financial PhraseBank” is a multiclass output instead having 604 (~12.46%) negative, 2879 (~59.41%) neutral and 1363 (~28.13%) positive data. On the other hand, “FiQA Task 1” consist of 1111 labelled data and its output is a continuous score ranged as [-1,1], where -1 represents totally negative and 1 represents totally positive, quantifying the degree of that sentiment. The mean, standard deviation, 1st quartile, median and 3rd quartile are 0.1226, 0.4033, -0.2605, 0.2480 and 0.4435 respectively. In the later section, we will call this combined dataset as “domain data”.

Sentiment140 (http://help.sentiment140.com/for-students)
Amazon reviews polarity dataset (https://www.kaggle.com/kritanjalijain/amazon-reviews?select=train.csv)
Financial PhraseBank (https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news)
FiQA Task 1 (https://sites.google.com/view/fiqa/home)

#### Architectures
All sentences were first passed through an embedding layer, where we use the glove-6B-300d pre-trained model, to convert each observation into a (300, Tx) matrix where Tx is the maximum number of words we limited to each sentence. 

Four architectures were experimented in this project and all of them are recurrent type neural networks. Their details are listed below ranked by architectural complexity:
1.	Bidirectional 1-layer RNN (RNN)
The first model consists of a single bi-directional simple recurrent layer with 64(x2) hidden units, with a total of 46,978 trainable parameters. 
2.	Bidirectional 1-layer LSTMs (LSTM)
Due to the Vanishing Gradient Problem of RNN layer, we also expect a LSTMs architecture would perform better in our case and thus, the second architecture are constructed by replacing the RNN cells by LSTM cells. As the validation accuracy is as high as the training accuracy, we further enhance the complexity by increasing the number of hidden units into 128(x2). Trainable parameters grow exponentially to 439,810.
3.	2-layers Bidirectional LSTMs (2l-LSTM)
We further improve the model complexity by doubling the LSTM layer and a batch normalization layer are added between layers. To avoid overfitting, a dropout layer with 0.3 are also added between layers and before the binary classification layer which result in 834, 562 trainable parameters.
4.	3-layers Bidirectional LSTMs (3l-LSTM)
Last by not least, the third LSTM layer with 64(x2) hidden units are added on top of 2l-LSTM resulting in 999,170 trainable parameters in total.

A dense layer with sigmoid function is added to each architecture to perform binary classification at the end. 

Glove (https://nlp.stanford.edu/projects/glove/)

Transfer learning Method
The exact approach to perform transfer learning always depends on 1) The amount of labelled domain data and 2) The variety of the domain data distribution from the streamline. In general, “Fine Tuning the final layer” is more appropriate when data are not sufficient data to avoid overfitting and “Transfer as initial” vice versa. In our project, both methods were experimented with.

#### Methodology
The entire workflow can be split into three parts. 1. Pre-processing, 2. Base Model Training and 3. Domain Adoption.

1.	Pre-processing 
i.	As the general data consists of tweets and product reviews, some sentence such as advertisements and non-English sentences were defined as noise data and therefore were removed at the very beginning. 
ii.	Some special patterns resulting from different language styling were removed as well. Examples include: “rt” – retweet, “@someone” – reply to someone, “&sth” – emoji and “www.sth.com” – website. Such patterns are removed from the original sentence. 
iii.	Non-alphabet symbols, such as | and ~ were removed, and duplicated formal punctuation were combined into a single occurrence from the sentence. i.e. “!!!!!!!!!” is replaced by a single “!”.
iv.	In sentiment analysis, there is no difference in meaning between upper-case wording and lower-case wording and hence, all sentences were converted to lower case.
v.	Stopword refers to the wording having no meaning for the current task and thus, a list of stop word were removed. In our project, we mainly rely on the stopword list from the nltk library and fine tune the exact list to be removed.
vi.	We also perform Lemmatization based on the WordNet PoS tagging and WordNetLemmatizer from the nltk library as well.
vii.	Through EDA, some sentences were discovered to be of extremely long length, thus sentences that exceed 130 words were classified as outliers and were removed.
viii.	Padding was performed to control the size of input data and input sentence length was limited to 50 words. All sentence less than 50 words will pad 0 which indicate nothing at the end of the sentences up to 50 dimensions. 
2.	Base Model Training
After processed data pre-processing, we have 5,573,908 general data in total where 50.038% are negative. 100,000 general data is randomly picked to act as testing data and 2% of remaining are used as validation data for model selection and the rest are training data.

All architectures (see Architectures) were trained with Adam optimizer using default learning rate -- 0.001, 8192 batch size and binary cross entropy as loss function. Model selections were also based on validation loss and the training were early stopped if 5 consecutive unimprovement existed or reach 50 epochs. 

3. Domain Adoption
The base models were then further trained on domain data in 2 different transfer learning method respectively. Based on different domain dataset, different treatment was made. Compared with the base model training, we split the domain data into 0.6/0.2/0.2 for training, validation, and testing purpose 

“Financial PhraseBank” has an imbalanced multiclass output (see Dataset). Thus, a class-weighted categorical cross entropy was used as loss function and the weights are defined as (number of sample in i-th class)/3. This is to force different classes to have equivalent importance toward the loss function. To match with this change, the last classification layer of experimented architectures is modified to use softmax function instead. The models were further trained with Adam optimizer using learning rate ranged from 2e−4 to 5e−3, as we would like to limit the maximum epochs up to 50 due to computational consideration, and 64 batch sizes. Again, the training was early stopped if 5 consecutive unimprovement existed or reach 50 epochs. For “Fine Tuning the final layer” method, all parameters from the base model will be untrainable except the last modified layer – the softmax classifier, while all parameters are free to move for “Transfer as initial”. 

“FiQA Task 1” consists of a linear output ranged between [-1, 1] (see Dataset). Thus, mean squared error was used as loss function and the last classification layer of experimented architectures is modified to a tanh function to match with this change. The models were further trained with Adam optimizer using learning rate ranged from 2e−3 to 5e−4 to limit the maximum epochs up to 50 due to computational consideration, and 32 batch sizes. Again, the training was early stopped if 5 consecutive unimprovement existed or reach 50 epochs. For “Fine Tuning the final layer” method, all parameters from the base model will be untrainable except the last modified layer – the tanh regressor, while all parameters are free to move for “Transfer as initial”.

#### Result
The general dataset has balanced label and accuracy and binary cross entropy were chosen as evaluation metrics on the base model. The results are as follow:
 
We achieve a 90.32% testing accuracy for using 3-layer-LSTM model on general sentiment analysis. As expected, the loss and accuracy improve accordingly with respect to the model complexity. The dropout layer still successfully controls potential overfitting problem. However, this obtained accuracy is less than our expectation because GloVe can perform as good as 99.x% in other research topics. The drop of accuracy may be attributed to inaccurate data cleaning, padding and architecture setting, etc. 
The domain data has a different label distribution from the base general data. For “Financial PhraseBank”, the categorical cross entropy loss is the main evaluation metric with the support of total accuracy and visualized confusion matrix. Although accuracy is not a good evaluation metric because of imbalanced labels, we are not interested in evaluating which model performing the best, but whether domain adoption is well performed using different transfer learning method separately. Thus, a visualized confusion matrix is a good enough tool to make direct comparison. For “FiQA Task 1”, the mean squared error works as the unique evaluation metric. The obtained results are as follow:
 
The final models on “Financial PhraseBank” achieve a 0.6679 testing loss and 74.43% testing accuracy using RNN model. In general, the final models on “Financial PhraseBank” work much better if we transfer the base model as initial weighting only. This indicates that a portion of domain adoption is performed during this transfer process when compared to the performance of fine tuning the last layer only. A serious domain adoption gap between general sentence and financial domain is also discovered. In addition, the performance discrepancy between training data and testing data increases with respect to the model complexity. This indicates that the problem of insufficient data which result in overfitting is magnified by increasing model complexity. The testing confusion metric below also echoes the same conclusion. The left figure is Fine Tuning RNN while transferred as initial RNN is on the right.
  
We achieved a testing loss of 0.1311 on “FiQA Task 1” using 2-layer-LSTM. This dataset is a much smaller dataset and only consist of 1111 data. As expected, models performing “Transfer as initial” were experienced a serious overfitting problem which performed worse, with higher testing MSE than the one employing “Transfer as initial”. If we inherit the conclusion from “Financial PhraseBank” that there is a huge domain adoption problem on FSA, these models do not perform any domain adoption.

#### Conclusion
We show the possibility of using transfer learning for tuning the task first, and then perform domain adoption. However, there is a huge gap between general sentence and financial text and thus, a sufficient large domain dataset is also required. Referenced to FinBERT, one may use another transfer learning framework to further perform unsupervised learning on financial corpora before fine tuning the task. As there is much more unlabeled financial text exist in this field, this approach should be under further consideration.

#### Licens
Inherit all the license form the used dataset and pre-trained model. 

#### Last Update
13 Aug 2021

