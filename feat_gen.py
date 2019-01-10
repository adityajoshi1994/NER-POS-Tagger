import pickle
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
class Lexicon: pass

class Cluster:
    def __init__(self):
        self.model = pickle.load(open('k_means.sav', 'rb'))
        self.wordVec = pickle.load(open('word_vec','rb'))

    def compute_cluster(self, word):
        if word in self.wordVec.wv.vocab:
            return str(self.model.predict(self.wordVec[word].reshape(1, -1))[0] + 1)
        else:
            return 0

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    pass
    
####### I have modified feats.py to pass lexicons dictionary to feat_gen.py #######
def token2features(sent, i, cluster, lexicons = {}, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")


    ftrs.append("STEM=" + stemmer.stem(word))
    ftrs.append("PREFIX1=" + word[0])
    ftrs.append("PREFIX2="+word[0:2])
    ftrs.append("SUFFIX1=" + word[-1])
    ftrs.append("SUFFIX2=" + word[-2:])
    ftrs.append("SUFFIX3=" + word[-3:])

    if(word==(word[0]+word[1:])):
        ftrs.append("IS_TITLE")
    if not word.isnumeric() and ((word.replace(".","",1)).isnumeric()):
        ftrs.append("IS_NUMERIC")
    if "@" in word:
         ftrs.append("HAS_@")
    if "#" in word:
         ftrs.append("HAS_#")
    if "*" in word:
         ftrs.append("HAS_*")
    if "!" in word:
         ftrs.append("HAS_INTERJECTION")
    if "?" in word:
        ftrs.append("HAS_?")
    if ".com" in word:
        ftrs.append("HAS_.com")

    if word in lexicons["names"]:
        ftrs.append("IS_NAME")
    if word in lexicons["location"]:
        ftrs.append("IS_LOCATION")
    if word in lexicons["stops"]:
        ftrs.append("IS_STOP")
    if word in lexicons["tvshow"]:
        ftrs.append("IS_TVSHOW")
    if word in lexicons["product"]:
        ftrs.append("IS_PRODUCT")
    if word in lexicons["company"]:
        ftrs.append("IS_COMPANY")
    if word in lexicons["sportsteam"]:
        ftrs.append("IS_TEAM")


    ftrs.append("CLUSTER" + str(int(cluster.compute_cluster(word))))

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, cluster, lexicons, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1,cluster, lexicons, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [["New", "York"]]
    lexfile = "file_path/data/lexicons.txt"
    with open(lexfile, "rb") as myFile:
        lexicons = pickle.load(myFile)
    preprocess_corpus(sents)
    cluster = Cluster()
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent,  i,cluster, lexicons)
