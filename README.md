# NER-POS-Tagger-backend

<h2>Structured Perceptron: </h2>
Structured perceptron uses a similar concept as a normal perceptron except that now it is used to learn a structure instead of 
a single prediction. So one can imagine it as an array of single perceptrons each capturing a specific part of the structure.
In structured perceptron the delta is calculated over this entire array instead of just the single predicted variable. 
The Named entity recognition and parts of speec tags exibit some structures and the model attempts to learn this from the twitter data.

<h2>Conditional Random Fields: </h2>
CRFs are graphical models similar to HMM(Hidden Markov Models) with the small difference that in CRFs we do global normalization instead of local normalizations as in the case of HMMs/MEMMs. This means that we are trying to assign a score to the entire sequences instead of just looking at the scores for the next word. This helps us capture the underlying context of a sentence better than what HMMs would capture.

eg: I fish often: This is a good example to understand the difference, the word fish in the sentence would most probably be labelled as a noun by the HMMs since it normalizes locally and only looks at the previous word and the current word whereas the CRFs would correctly classify it as a verb since it looks at the entire sequence of the sentence.

CRFs use dynamic programming to get the label sequences with maximum score. This form of inference is also called Viterbi decoding.

<h2> Viterbi Decoding: </h2>
Viterbi decoding is a way by means of which we can know the probablity distributions of sequence of labels Y for a given sentence i.e P(Y|X) where Y and X are both sequences. The key fact that the algorithm uses is that if there are two functions f(a) >= 0 and g(a, b) >= 0 then maxOver(a, b) f(a)g(a, b) = maxOver(a){f(a)maxOver(b)g(a, b)}

The following recurrence relation ultimately helps us to get the maximum scores
R(i, yi) = maxOverYi-1(e(xi|yi) * t(yi|yi - 1) * R(i - 1,yi - 1))

Run time complexity for the algorithm: O(NL^2) where N is the #tokens and L is the #Labels.
