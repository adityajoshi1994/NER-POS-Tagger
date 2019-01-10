import pickle
import timeit
from flask import Flask, request, abort, jsonify
import json
# start = timeit.timeit()
# tagger = pickle.load(open('finalized_model_CRF_ner.sav','rb'))
# print tagger.tag_sent(["I","live","in","America"])
# end = timeit.timeit()
# print "Time to run: ", (end - start)

NERTagger = pickle.load(open('finalized_model_CRF_ner.sav', 'rb'))
POSTagger = pickle.load(open('finalized_model_CRF_pos.sav', 'rb'))
app = Flask(__name__)

@app.route('/api/gettags', methods=['GET'])
def getTags():
    sentence = request.args.get('input')
    print sentence
    sentenceArray = sentence.strip().split()
    NER = NERTagger.tag_sent(sentenceArray)
    POS = POSTagger.tag_sent(sentenceArray)
    data = {'NER': NER.tolist(), 'POS': POS.tolist()}
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)




