

# An Intuitive Understanding of Word Embedding's: From Count Vectors to Word2Vec

Sure, a computer can match two strings and tell you whether they are same or not. But how do we make computers tell you about football or Ronaldo when you search for Messi? How do you make a computer understand that “Apple” in “Apple is a tasty fruit” is a fruit that can be eaten and not a company?

The answer to the above questions lie in creating a representation for words that capture their *meanings*, *semantic relationships* and the different types of contexts they are used in.

And all of these are implemented by using Word Embedding's or numerical representations of texts so that computers may handle them.



### 1. What are Word Embedding's?

In very simplistic terms, Word Embedding's are the texts converted into numbers and there may be different numerical representations of the same text.

*Why do we need Word Embedding's?*

Many Machine Learning algorithms and almost all Deep Learning Architectures are incapable of processing *strings* or *plain text* in their raw form.

They require numbers as inputs to perform any sort of job, be it classification, regression etc. in broad terms.



Some real world applications of text applications are – sentiment analysis of reviews by Amazon etc., document or news classification or clustering by Google etc.



Let us now define Word Embedding's formally. A Word Embedding format generally tries to map a word using a dictionary to a vector. Let us break this sentence down into finer details to have a clear view.



Take a look at this example – **sentence**=” Word Embedding's are Word converted into numbers ”

A *word* n this **sentence** may be “Embedding's” or “numbers ” etc.

A *dictionary* may be the list of all unique words in the **sentence. **So, a dictionary may look like – [‘Word’,’Embedding's’,’are’,’Converted’,’into’,’numbers’]



A **vector** representation of a word may be a one-hot encoded vector where 1 stands for the position where the word exists and 0 everywhere else. The vector representation of “numbers” in this format according to the above dictionary is [0,0,0,0,0,1] and of converted is[0,0,0,1,0,0].



## 2. Different types of Word Embedding's

The different types of word embeddings can be broadly classified into two categories-

1. Frequency based Embedding
2. Prediction based Embedding

Let us try to understand each of these methods in detail.

 

### 2.1 Frequency based Embedding

There are generally three types of vectors that we encounter under this category.

1. Count Vector
2. TF-IDF Vector
3. Co-Occurrence Vector

Let us look into each of these vectorization methods in detail.

 

#### 2.1.1 Count Vector

Consider a Corpus C of D documents {d1,d2…..dD} and N unique tokens extracted out of the corpus C. The N tokens will form our dictionary and the size of the Count Vector matrix M will be given by D X N. Each row in the matrix M contains the frequency of tokens in document D(i).

Let us understand this using a simple example.

D1: He is a lazy boy. She is also lazy.

D2: Neeraj is a lazy person.

The dictionary created may be a list of unique tokens(words) in the corpus =[‘He’,’She’,’lazy’,’boy’,’Neeraj’,’person’]

Here, D=2, N=6

The count matrix M of size 2 X 6 will be represented as –

|      | He   | She  | lazy | boy  | Neeraj | person |
| ---- | ---- | ---- | ---- | ---- | ------ | ------ |
| D1   | 1    | 1    | 2    | 1    | 0      | 0      |
| D2   | 0    | 0    | 1    | 0    | 1      | 1      |

Now, a column can also be understood as word vector for the corresponding word in the matrix M. For example, the word vector for ‘lazy’ in the above matrix is [2,1] and so on.Here, the *rows* correspond to the *documents* in the corpus and the *columns* correspond to the *tokens* in the dictionary. The second row in the above matrix may be read as – D2 contains ‘lazy’: once, ‘Neeraj’: once and ‘person’ once.

Now there may be quite a few variations while preparing the above matrix M. The variations will be generally in-

1. The way dictionary is prepared.
   Why? Because in real world applications we might have a corpus which contains millions of documents. And with millions of document, we can extract hundreds of millions of unique words. So basically, the matrix that will be prepared like above will be a very sparse one and inefficient for any computation. So an alternative to using every unique word as a dictionary element would be to pick say top 10,000 words based on frequency and then prepare a dictionary.
2. The way count is taken for each word.
   We may either take the frequency (number of times a word has appeared in the document) or the presence(has the word appeared in the document?) to be the entry in the count matrix M. But generally, frequency method is preferred over the latter.

Below is a representational image of the matrix M for easy understanding.

 

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/04164920/count-vector.png)

## 2.2 Prediction based Vector



**Pre-requisite** : Knowledge in neural networks

So far, we have seen deterministic methods to determine word vectors. But these methods proved to be limited in their word representations until Mitolov etc. el introduced word2vec to the NLP community. 

These methods were prediction based in the sense that they provided probabilities to the words and proved to be state of the art for tasks like word analogies and word similarities. They were also able to achieve tasks like King -man +woman = Queen, which was considered a result almost magical. So let us look at the word2vec model used as of today to generate word vectors.



Word2vec is not a single algorithm but a combination of two techniques – CBOW(Continuous bag of words) and Skip-gram model. Both of these are shallow neural networks which map word(s) to the target variable which is also a word(s). Both of these techniques learn weights which act as word vector representations. Let us discuss both these methods separately and gain intuition into their working.

### 2.2.1 CBOW (Continuous Bag of words)

The way CBOW work is that it tends to predict the probability of a word given a context. A context may be a single word or a group of words. But for simplicity, I will take a single context word and try to predict a single target word.

Suppose, we have a corpus C = “Hey, this is sample corpus using only one context word.” and we have defined a context window of 1. This corpus may be converted into a training set for a CBOW model as follow. The input is shown below. The matrix on the right in the below image contains the one-hot encoded from of the input on the left.

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/04205949/cbow1.png)

The target for a single datapoint say Datapoint 4 is shown as below

| Hey  | this | is   | sample | corpus | using | only | one  | context | word |
| ---- | ---- | ---- | ------ | ------ | ----- | ---- | ---- | ------- | ---- |
| 0    | 0    | 0    | 1      | 0      | 0     | 0    | 0    | 0       | 0    |

 

This matrix shown in the above image is sent into a shallow neural network with three layers: an input layer, a hidden layer and an output layer. The output layer is a softmax layer which is used to sum the probabilities obtained in the output layer to 1. Now let us see how the forward propagation will work to calculate the hidden layer activation.

Let us first see a diagrammatic representation of the CBOW model.

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/04224109/Screenshot-from-2017-06-04-22-40-29.png)

The matrix representation of the above image for a single data point is below.

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/04222108/Screenshot-from-2017-06-04-22-19-202.png)

The flow is as follows:

1. The input layer and the target, both are one- hot encoded of size [1 X V]. Here V=10 in the above example.
2. There are two sets of weights. one is between the input and the hidden layer and second between hidden and output layer.
   Input-Hidden layer matrix size =[V X N] , hidden-Output layer matrix  size =[N X V] : Where N is the number of dimensions we choose to represent our word in. It is arbitary and a hyper-parameter for a Neural Network. Also, N is the number of neurons in the hidden layer. Here, N=4.
3. There is a no activation function between any layers.( and by no, i mean linear activation)
4. The input is multiplied by the input-hidden weights and called hidden activation. It is simply the corresponding row in the input-hidden matrix copied.
5. The hidden input gets multiplied by hidden- output weights and output is calculated.
6. Error between output and target is calculated and propagated back to re-adjust the weights.
7. The weight  between the hidden layer and the output layer is taken as the word vector representation of the word.

We saw the above steps for a single context word. Now, what about if we have multiple context words? The image below describes the architecture for multiple context words.

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/04220606/Screenshot-from-2017-06-04-22-05-44-261x300.png)

Below is a matrix representation of the above architecture for an easy understanding.

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/04221550/Screenshot-from-2017-06-04-22-14-311.png)

The image above takes 3 context words and predicts the probability of a target word. The input can be assumed as taking three one-hot encoded vectors in the input layer as shown above in red, blue and green.

So, the input layer will have 3 [1 X V] Vectors in the input as shown above and 1 [1 X V] in the output layer. Rest of the architecture is same as for a 1-context CBOW.

The steps remain the same, only the calculation of hidden activation changes. Instead of just copying the corresponding rows of the input-hidden weight matrix to the hidden layer, an average is taken over all the corresponding rows of the matrix. We can understand this with the above figure. The average vector calculated becomes the hidden activation. So, if we have three context words for a single target word, we will have three initial hidden activations which are then averaged element-wise to obtain the final activation.

In both a single context word and multiple context word, I have shown the images till the calculation of the hidden activations since this is the part where CBOW differs from a simple MLP network. The steps after the calculation of hidden layer are same as that of the MLP as mentioned in this article – [Understanding and Coding Neural Networks from scratch](https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/).

The differences between MLP and CBOW are  mentioned below for clarification:

1. The objective function in MLP is a MSE(mean square error) whereas in CBOW it is negative log likelihood of a word given a set of context i.e -log(p(wo/wi)), where p(wo/wi) is given as

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/04230048/AAEAAQAAAAAAAA18AAAAJGNkMGYxMDIxLWY5NjgtNGEzMy1hMjAyLWU4MmI4ZWUwNDNhYw-300x91.jpg)

wo : output word
wi: context words

\2. The gradient of error with respect to hidden-output weights and input-hidden weights are different since MLP has  sigmoid activations(generally) but CBOW has linear activations. The method however to calculate the gradient is same as an MLP.

 

**Advantages of CBOW:**

1. Being probabilistic is nature, it is supposed to perform superior to deterministic methods(generally).
2. It is low on memory. It does not need to have huge RAM requirements like that of co-occurrence matrix where it needs to store three huge matrices.

 

**Disadvantages of CBOW:**

1. CBOW takes the average of the context of a word (as seen above in calculation of hidden activation). For example, Apple can be both a fruit and a company but CBOW takes an average of both the contexts and places it in between a cluster for fruits and companies.
2. Training a CBOW from scratch can take forever if not properly optimized.



## 2.2.2 Skip – Gram model

Skip – gram follows the same topology as of CBOW. It just flips CBOW’s architecture on its head. The aim of skip-gram is to predict the context given a word. Let us take the same corpus that we built our CBOW model on. C=”Hey, this is sample corpus using only one context word.” Let us construct the training data.

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/04235354/Capture1-300x222.png)

The input vector for skip-gram is going to be similar to a 1-context CBOW model. Also, the calculations up to hidden layer activations are going to be the same. The difference will be in the target variable. Since we have defined a context window of 1 on both the sides, there will be “**two” one hot encoded target variables** and “**two” corresponding outputs** as can be seen by the blue section in the image.

Two separate errors are calculated with respect to the two target variables and the two error vectors obtained are added element-wise to obtain a final error vector which is propagated back to update the weights.

The weights between the input and the hidden layer are taken as the word vector representation after training. The loss function or the objective is of the same type as of the CBOW model.

The skip-gram architecture is shown below.

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/05000515/Capture2-276x300.png)

 

For a better understanding, matrix style structure with calculation has been shown below.

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/05122225/skip.png)

 

Let us break down the above image.

Input layer  size – [1 X V], Input hidden weight matrix size – [V X N], Number of neurons in hidden layer – N, Hidden-Output weight matrix size – [N X V], Output layer size – C [1 X V]

In the above example, C is the number of context words=2, V= 10, N=4

1. The row in red is the hidden activation corresponding to the input one-hot encoded vector. It is basically the corresponding row of input-hidden matrix copied.
2. The yellow matrix is the weight between the hidden layer and the output layer.
3. The blue matrix is obtained by the matrix multiplication of hidden activation and the hidden output weights. There will be two rows calculated for two target(context) words.
4. Each row of the blue matrix is converted into its softmax probabilities individually as shown in the green box.
5. The grey matrix contains the one hot encoded vectors of the two context words(target).
6. Error is calculated by substracting the first row of the grey matrix(target) from the first row of the green matrix(output) element-wise. This is repeated for the next row. Therefore, for **n **target context words, we will have **n **error vectors.
7. Element-wise sum is taken over all the error vectors to obtain a final error vector.
8. This error vector is propagated back to update the weights.

### Advantages of Skip-Gram Model

1. Skip-gram model can capture two semantics for a single word. i.e it will have two vector representations of Apple. One for the company and other for the fruit.
2. Skip-gram with negative sub-sampling outperforms every other method generally.

 

[This](http://bit.ly/wevi-online) is an excellent interactive tool to visualize CBOW and skip gram in action. I would suggest you to really go through this link for a better understanding.



## 3. Word Embeddings use case scenarios

Since word embeddings or word Vectors are numerical representations of contextual similarities between words, they can be manipulated and made to perform amazing tasks like-

1. Finding the degree of similarity between two words.
   `model.similarity('woman','man')`
   `0.73723527`
2. Finding odd one out.
   `model.doesnt_match('breakfast cereal dinner lunch';.split())`
   `'cereal'`
3. Amazing things like woman+king-man =queen
   `model.most_similar(positive=['woman','king'],negative=['man'],topn=1)`
   `queen: 0.508`
4. Probability of a text under the model
   `model.score(['The fox jumped over the lazy dog'.split()])`
   `0.21`

Below is one interesting visualisation of word2vec.

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/05003425/graph1-300x277.jpg)

The above image is a t-SNE representation of word vectors in 2 dimension and you can see that two contexts of apple have been captured. One is a fruit and the other company.

\5.  It can be used to perform Machine Translation.
![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/05003807/ml-300x211.png)

The above graph is a bilingual embedding with chinese in green and english in yellow. If we know the words having similar meanings in chinese and english, the above bilingual embedding can be used to translate one language into the other.

## 4. Using pre-trained word vectors

We are going to use google’s pre-trained model. It contains word vectors for a vocabulary of 3 million words trained on around 100 billion words from the google news dataset. The downlaod link for the model is [this](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). Beware it is a 1.5 GB download.

```python
from gensim.models import Word2Vec

#loading the downloaded model

model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True,norm_only=True)

#the model is loaded. It can be used to perform all of the tasks mentioned above.

# getting word vectors of a word

dog = model['dog']

#performing king queen magic

print(model.most_similar(positive=['woman', 'king'], negative=['man']))

#picking odd one out

print(model.doesnt_match("breakfast cereal dinner lunch".split()))

#printing similarity index

print(model.similarity('woman', 'man'))

```



## 5. Training your own word vectors

We will be training our own word2vec on a custom corpus. For training the model we will be using gensim and the steps are illustrated as below.

word2Vec requires that a format of list of list for training where every document is contained in a list and every list contains list of tokens of that documents. I won’t be covering the pre-preprocessing part here. So let’s take an example list of list to train our word2vec model.



```python
from gensim.models import KeyedVectors
from gensim.models import word2vec
import logging
import os

sentence=[['Neeraj','Boy'],['Sarwan','is'],['good','boy']]

#training word2vec on 3 sentences

model = word2vec.Word2Vec(sentence, min_count=1,size=300,workers=4)

#using the model#The new trained model can be used similar to the pre-trained ones.

#printing similarity index

print(model.similarity('Boy', 'Neeraj'))

```

Let us try to understand the parameters of this model.

sentence – list of list of our corpus
min_count=1 -the threshold value for the words. Word with frequency greater than this only are going to be included into the model.
size=300 – the number of dimensions in which we wish to represent our word. This is the size of the word vector.
workers=4 – used for parallelization

***

# Using pre-trained embedding's

In general, you will train your own word2vec or GloVe model from scratch only if you have a very
large amount of very specialized text. By far the most common use case for Embedding's is to use pre-trained.
embedding's in some way in your network. The three main ways in which you would use
embeddings in your network are as follows:

- Learn embeddings from scratch
- Fine-tune learned embeddings from pre-trained GloVe/word2vec models
- Look up embedding's from pre-trained GloVe/word2vec models



##### 1. Learn embedding's from scratch

In the first option, the embedding weights are initialized to small random values and trained using
backpropagation. You saw this in the examples for skip-gram and CBOW models in Keras. This is the
default mode when you use a Keras Embedding layer in your network.

##### 2. Fine-tune learned embeddings from pre-trained GloVe/word2vec models 

In the second option, you build a weight matrix from a pre-trained model and initialize the weights of
your embedding layer with this weight matrix. The network will update these weights using
backpropagation, but the model will converge faster because of good starting weights.

##### 3. Look up embedding's from pre-trained GloVe/word2vec models 

The third option is to look up word embeddings from a pre-trained model, and transform your input to
embedded vectors. You can then train any machine learning model (that is, not necessarily even a
deep learning network) on the transformed data. If the pre-trained model is trained on a similar
domain as the target domain, this usually works very well and is the least expensive option.



For general use with English language text, you can use Google's word2vec model trained over 10
billion words from the Google news dataset. The vocabulary size is about 3 million words and the
dimensionality of the embedding is 300. The Google news model (about 1.5 GB) can be downloaded
from here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing.

Similarly, a pre-trained model trained on 6 billion tokens from English Wikipedia and the giga-word
corpus can be downloaded from the GloVe site. The vocabulary size is about 400,000 words and the
download provides vectors with dimensions 50, 100, 200, and 300. The model size is about 822 MB.
Here is the direct download URL (http://nlp.stanford.edu/data/glove.6B.zip) for this model. Larger models based
on the Common Crawl and Twitter are also available from the same location.

In the following sections, we will look at how to use these pre-trained models in the three ways
listed.



### 1. Learn embeddings from scratch

In the first option, the embedding weights are initialized to small random values and trained using
backpropagation. You saw this in the examples for skip-gram and CBOW models in Keras. This is the
default mode when you use a Keras Embedding layer in your network.



In this example, we will train a one-dimensional convolutional neural network (CNN) to classify
sentences as either positive or negative. Recall that CNNs exploit spatial
structure in images by enforcing local connectivity between neurons of adjacent layers.



In our example network, the input text is converted to a sequence of word indices. Note that we have
used the natural language toolkit (NLTK) to parse the text into sentences and words. We could also
have used regular expressions to do this, but the statistical models supplied by NLTK are more
powerful at parsing than regular expressions.



The sequence of word indices is fed into an array of embedding layers of a set size (in our case, the
number of words in the longest sentence). The embedding layer is initialized by default to random
values. The output of the embedding layer is connected to a 1D convolutional layer that convolves (in
our example) word trigrams in 256 different ways (essentially, it applies different learned linear
combinations of weights on the word embeddings). These features are then pooled into a single
pooled word by a global max pooling layer. This vector (256) is then input to a dense layer, which
outputs a vector (2). A softmax activation will return a pair of probabilities, one corresponding to
positive sentiment and another corresponding to negative sentiment. The network is shown in the
following figure:

### How is the validation split computed?

If you set the `validation_split` argument in `model.fit` to e.g. 0.1, then the validation data used will be the *last 10%* of the data. If you set it to 0.25, it will be the last 25% of the data, etc. Note that the data isn't shuffled before extracting the validation split, so the validation is literally just the *last* x% of samples in the input you passed.

The same validation set is used for all epochs (within a same call to `fit`).

### Embedding

```
keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)

```

Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

This layer can only be used as the first layer in a model.

**Example**

```
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)

```

**Arguments**

- **input_dim**: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
- **output_dim**: int >= 0. Dimension of the dense embedding.
- **embeddings_initializer**: Initializer for the `embeddings` matrix (see [initializers](https://keras.io/initializers/)).
- **embeddings_regularizer**: Regularizer function applied to the `embeddings` matrix (see [regularizer](https://keras.io/regularizers/)).
- **embeddings_constraint**: Constraint function applied to the `embeddings` matrix (see [constraints](https://keras.io/constraints/)).
- **mask_zero**: Whether or not the input value 0 is a special "padding" value that should be masked out. This is useful when using [recurrent layers](https://keras.io/layers/recurrent/) which may take variable length input. If this is `True` then all subsequent layers in the model need to support masking or an exception will be raised. If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary (input_dim should equal size of vocabulary + 1).
- **input_length**: Length of input sequences, when it is constant. This argument is required if you are going to connect `Flatten`then `Dense` layers upstream (without it, the shape of the dense outputs cannot be computed).

**Input shape**

2D tensor with shape: `(batch_size, sequence_length)`.

**Output shape**

3D tensor with shape: `(batch_size, sequence_length, output_dim)`.





























































































***

# Word Embedding

A way to represent the words in the text to numerical vectors. So they can be analyzed by standard machine learning algorithms that require vectors as numerical input.

For Example:

​	**One Hot Encoding:**  It is a vector which is 0 in most dimensions and 1 in a single dimension.

​	A major disadvantage with one-hot encoding is there is no way to represent similarity between the words.

​	In any given corpus(large and structured set of texts) you would expect words such as (cat,dog), (knife,spoon) and so on to have some similarity.

​	Similarity between the vectors is computed using dot product, which is the sum of element wise multiplication between vector elements.c

​	In case of 1 hot encoded vectors, the dot product between any tow words in a corpus(large and structured set of texts) is always zero. Hence we cannot use 1 hot encoding in this type of situation.

​	To overcome the limitations of one-hot encoding, the NLP community has borrowed techniques from
information retrieval (IR) to vectorize text using the document as the context. Notable techniques are TF-IDF and latent semantic analysis.

***

Here in this part we are going to learn about,

- Glove
- word2vec

Collectively known as the distributed representations of words.

We will learn  how to generate your own embedding in keras code, as well as how to use and fine-tune pre-trained word2vec and GloVe models.

***

### Distributed representation

Attempts to capture the meanings of a word by considering its relations with other words in its context.

Consider the following pair of sentence,

*Paris is the capital of France*

*Berlin is the capital of Germany*

Even assuming you have no knowledge of world geography (or English for that matter), you would
still conclude without too much effort that the word pairs (Paris, Berlin) and (France, Germany)
were related in some way, and that corresponding words in each pair were related in the same way to
each other, that is:

Paris : France :: Berlin : Germany



Distributed representation aims to convert word to vectors where the similarity between vectors correlate with the semantic similarity between words.

***

### word2vec

It is a unsupervised model taking input as large corpus(large and structured set of texts) of text and producing a vector space of words.

***word2vec vs one hot encoding***

- word2vec dimensionality is lower when compared to one hot encoding, which is the size of vocabulary.
- word2vec embedding space is more dense compared to embedding space of 1 hot encoding.

2 architecture for word2vec are:

- continuous bag of words(CBOW)
- skip-gram

**CBOW architecture**-> predicts current word  given surrounding words. Here word assumption does not influence prediction.

**skip-gram architecture**->Model predicts surrounding words given the center word.

CBOW is faster but skip-gram does a better job in predicting the infrequent words.

***

### The skip-gram word2vec model

The skip-gram model is trained to predict the surrounding words given the current word.

Example:

*I love green eggs and ham.*

Assuming window size = 3, this sentence can be broken down into following sets of **(context,word)** pairs:

​					([I, green], love)
​					([love, eggs], green)
​					([green, and], eggs)
​					...

Since skip-gram model predicts a context word given the center word, we can convert the predicting dataset to one of **(input,output) pairs** .That is, given an input word, we expect the skip-gram
model to predict the output word:

*(love, I), (love, green), (green, love), (green, eggs), (eggs, green), (eggs, and), ...*

We can also generate additional negative samples by pairing each input word with some random words in the vocabulary.

Example:

*(love, Sam), (love, zebra), (green, thing), ...*

Finally, we generate positive and negative examples for our classifier:

*((love, I), 1), ((love, green), 1), ..., ((love, Sam), 0), ((love, zebra), 0), ...*

We can now train a classifier that takes in a word vector and a context vector and learns to predict
one or zero depending on whether it sees a positive or negative sample.

**To Note:**

***window_size*** : *int maximum distance between two words in a positive couple.*

***

### The CBOW word2vec model

CBOW model predicts the center word given the context word.

if text is "I love green eggs and ham ."

Thus in the first tuple, CBOW model needs to predict the output word love, given the context word I & green:

([I,green],love), ([love,eggs],green), ([green,and],eggs)

Like skip-gram model, the CBOW-model is as a classifier that takes context word as input and predicts the target word.





The input to the model is the word IDs for the context words.

- These word IDs are fed into a common **embedding layer** that is initialized with small random weights.

- Each word IDs transformed into vector of size(embed_size) by the embedding layer.

- Each row of the input context is transformed into matrix of size(2*window_size, embed_size) by this layer.

- This is then fed into a lambda layer, which computes average of all embedding.

- This is then fed into a lambda layer, which computes an average of all the embedding's.

- This average is then fed to a dense layer,which creates a dense vector of size (vocab_size) for each row.

- The activation function on the dense layer is a softmax, which reports the maximum value on the output vector as a probability. The ID with the maximum probability corresponds to the target word.

  ​             

     

  ​

  ***

  ​

  ### GloVe (Global Vectors for Word Representation)

  Training is performed on aggregated global word-word
  co-occurrence statistics from a corpus(large and structured set of texts), and the resulting representations showcase interesting
  linear substructures of the word vector space.

  GloVe -> Count-Based Model

  word2vec -> Predictive Model

  ​

***





