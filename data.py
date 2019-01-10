#!/bin/python
import pickle
def read_twitter(dname="pos"):
    """Read the twitter train, dev, and test data from the default location.

    The returned object contains {train, test, dev}_{sents, labels}.
    """
    class Data: pass
    data = Data()
    # training data
    data.train_sents, data.train_labels = read_file("file_path_train" + dname)
    data.dev_sents, data.dev_labels = read_file("file_path_dev." + dname)
    # test data
    # Following is commented, only useful once test data is available.
    #data.test_sents, data.test_labels = read_file("data/twitter_test." + dname)
    # print statistics
    print "Twitter %s data loaded." % dname
    print ".. # train sents", len(data.train_sents)
    print ".. # dev sents", len(data.dev_sents)
    #print ".. # test sents", len(data.test_sents)
    return data

def read_file(filename):
    """Read the file in CONLL format, assumes one token and label per line."""
    sents = []
    labels = []
    with open(filename, 'r') as f:
        curr_sent = []
        curr_labels = []
        label_set = set()
        for line in f.readlines():
            if len(line.strip()) == 0:
                # sometimes there are empty sentences?
                if len(curr_sent) != 0:
                    # end of sentence
                    sents.append(curr_sent)
                    labels.append(curr_labels)
                    curr_sent = []
                    curr_labels = []
            else:
                token, label = line.split()
                curr_sent.append(unicode(token, 'utf-8'))
                curr_labels.append(label)
                label_set.add(label)
        print(label_set)
    return sents, labels

def write_preds(fname, sents, labels, preds):
    """Writes the output of a sentence in CONLL format, including predictions."""
    f = open(fname, "w")
    assert len(sents) == len(labels)
    assert len(sents) == len(preds)
    for i in xrange(len(sents)):
        write_sent(f, sents[i], labels[i], preds[i])
    f.close()

def write_sent(f, toks, labels, pred = None):
    """Writes the output of a sentence in CONLL format, including predictions (if pred is not None)"""
    for i in xrange(len(toks)):
        f.write(toks[i].encode('utf-8') + "\t" + labels[i])
        if pred is not None:
            f.write("\t" + pred[i])
        f.write("\n")
    f.write("\n")

def file_splitter(all_file, train_file, dev_file):
    """Splits the labeled data into train and dev, sentence-wise."""
    import random
    all_sents, all_labels = read_file(all_file)
    train_f = open(train_file, "w")
    dev_f = open(dev_file, "w")
    seed = 0
    dev_prop = 0.25
    rnd = random.Random(seed)
    for i in xrange(len(all_sents)):
        if rnd.random() < dev_prop:
            write_sent(dev_f, all_sents[i], all_labels[i])
        else:
            write_sent(train_f, all_sents[i], all_labels[i])
    train_f.close()
    dev_f.close()

def synthetic_data():
    """A very simple, three sentence dataset, that tests some generalization."""
    class Data: pass
    data = Data()
    data.train_sents = [
        [ "Obama", "is", "awesome" , "."],
        [ "Michelle", "is", "also", "awesome" , "."],
        [ "Awesome", "is", "Obama", "and", "Michelle", "."]
    ]
    data.train_labels = [
        [ "PER", "O", "ADJ" , "END"],
        [ "PER", "O", "O", "ADJ" , "END"],
        [ "ADJ", "O", "PER", "O", "PER", "END"]
    ]
    data.dev_sents = [
        [ "Michelle", "is", "awesome" , "."],
        [ "Obama", "is", "also", "awesome" , "."],
        [ "Good", "is", "Michelle", "and", "Obama", "."]
    ]
    data.dev_labels = [
        [ "PER", "O", "ADJ" , "END"],
        [ "PER", "O", "O", "ADJ" , "END"],
        [ "ADJ", "O", "PER", "O", "PER", "END"]
    ]
    return data

def get_dict(files):
    dict = {}
    sents = []
    for filename in files:
        with open(filename, 'r') as f:
            curr_sent = []
            for line in f.readlines():
                curr_sent = line.split()
                for i in range(min(10000,len(curr_sent)) ):
                    dict[curr_sent[i]] = True
    return dict

def add_lexicons(lexicon):
    lexicon.dict = {}
    basedir = "file_path/data/lexicon/"
    firstnames = basedir + "firstname.500"
    lastnames = basedir + "lastname.500"
    stops = basedir+"english.stop"
    clocations = basedir + "location.country"
    platforms = basedir+ "cvg.cvg_platform"
    tvshows = basedir + "tv.tv_program"
    brands = basedir + "business.brand"
    consumers = basedir + "business.consumer_company"
    sports = basedir + "sports.sports_team"
    channel = basedir + "broadcast.tv_channel"
    sponsor = basedir + "business.sponsor"
    product = basedir + "product"
    #locations = basedir + "location"

    lexicon.dict["names"] = get_dict([firstnames,lastnames])
    lexicon.dict["location"] = get_dict([clocations])
    lexicon.dict["stops"] = get_dict([stops])
    lexicon.dict["tvshow"] = get_dict([tvshows])
    lexicon.dict["product"] = get_dict([platforms, product])
    lexicon.dict["company"] = get_dict([brands, consumers, channel, sponsor])
    lexicon.dict["sportsteam"] = get_dict([sports])

    lexicon.name="aditya"

if __name__ == "__main__":
    # Do no run, the following function was used to generate the splits
    # file_splitter("data/twitter_train_all.pos", "data/twitter_train.pos", "data/twitter_dev.pos")

    # code to read lexicon files and store resulting dictionary in a lexicon.txt file
    # class Lexicon: pass
    # lexicon = Lexicon()
    # add_lexicons(lexicon)
    # lexicon_out = "/Users/adityajoshi/UCI/Stats NLP/data/lexicons.txt"
    # with open(lexicon_out, "wb") as myFile:
    #      pickle.dump(lexicon.dict, myFile)
    # print (lexicon.name)
    # Train the tagger

    dname = "ner"
    #dname = "ner"
    data = read_twitter(dname)
    # data = synthetic_data()

    C = [0.125, 0.5, 1, 2, 8, 32, 128, 256, 1024]
    tag = 'Logistic'
    import tagger
    if tag == 'Logistic':
        tagger = tagger.LogisticRegressionTagger()
    elif tag == 'CRF':
        tagger = tagger.CRFPerceptron()

    tagger.fit_data(data.train_sents, data.train_labels)
    # Evaluation (also writes out predictions)
    print "### Train evaluation"
    data.train_preds = tagger.evaluate_data(data.train_sents, data.train_labels)
    write_preds("file_path_twitter_train.%s.pred" % dname,
       data.train_sents, data.train_labels, data.train_preds)
    print "### Dev evaluation"
    data.dev_preds = tagger.evaluate_data(data.dev_sents, data.dev_labels)
    write_preds("file_Path_twitter_dev.%s.pred" % dname,
       data.dev_sents, data.dev_labels, data.dev_preds)

    # Following is commented, only useful once test data is available.
    # print "### Test evaluation"
    # data.test_preds = tagger.evaluate_data(data.test_sents, data.test_labels)
    # write_preds("/Users/adityajoshi/UCI/Stats NLP/data/twitter_test.%s.pred" % dname,
    #    data.test_sents, data.test_labels, data.test_preds)

    # filename = 'finalized_model_' + tag + '_' + dname + '.sav'
    # pickle.dump(tagger, open(filename, 'wb'))
    print 'Trying this out', tagger.tag_sent(["I", "live", "in", "America"])