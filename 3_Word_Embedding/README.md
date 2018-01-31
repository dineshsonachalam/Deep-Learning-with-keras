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

  ***

  ​