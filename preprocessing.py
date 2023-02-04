import conllu # reading ConLL-U files
import copy # deep-copying Tokens and and TokenLists
#import os
#os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
from conlluutils import *


# This function is a hack,
# but it changes any "flat" dependency relations
# with head a NOUN and dependent a PROPN
# into "nmod"...
# or into "flat" dependent on a previous PROPN
# if there are multiple on the same NOUN
# ... it just makes the denotations come out nicer.
# It takes a TokenList as input,
# and returns a new TokenList with the change.
def switch_propnflattonmod(sentence):
    sentence = copy.deepcopy(sentence)
    seenheads = {}
    for token in sentence:
        if token['upos'] == 'PROPN' and token['deprel'] == 'flat':
            if sentence.filter(id=token['head'])[0]['upos'] == 'NOUN':
                if token['head'] in seenheads.keys():
                    token['head'] = seenheads[token['head']]
                else:
                    token['deprel'] = 'nmod'
                    seenheads[token['head']] = token['id']
                    token['upos'] = 'PROPN-MOD'
    return conllu.TokenList(sentence)

# This function takes a TokenList as input
# and returns a new TokenList,
# very similar to the first,
# but with null definite determiners inserted just before proper nouns without existing determiners,
# and null indefinite determiners inserted just before common nouns without existing determiners.
# Because of a hack involving the "flat" relation,
# no determiner is inserted if the noun is related to its head with the "amod" relation.
def add_nulldeterminers(sentence):
    sentence = copy.deepcopy(sentence)
    determinered = [sentence.filter(id=token['head'])[0] for token in sentence
                        if (token['upos']=='DET' and token['deprel']=='det')]
    determinerless = [token for token in sentence
                        if (token['upos'] in ('NOUN','PROPN','PRON')
                            and token not in determinered
                            and token['deprel'] != "amod")]
    for token in determinerless:
        sentence.append(conllu.models.Token({'id':token['id']-0.5,
                            'form':'0def' if token['upos'] in ('PROPN','PRON') else '0indf',
                            'lemma':'0def' if token['upos'] in ('PROPN','PRON') else '0indf',
                            'upos':'DET',
                            'xpos':None,
                            'feats':None,
                            'head':token['id'],
                            'deprel':'det',
                            'deps': None,
                            'misc': None
                        }))
    sentence = conllu.TokenList(sorted(sentence,key=lambda x:x['id']))
    return reindex_tokenlist(sentence)

# This combines all the previous steps in the correct order.
# So, it changes flat relations between Nouns and Proper Nouns to "amod" relations (for semantic reasons),
# flattens remaining "flat" relations,
# flattens remaining "compound" relations,
# and adds null determiners to any determiner-less nouns and proper nouns (except those modified by "amod"),
# returning a new TokenList
def preprocess(sentence):
    flat = flatten_relation_list(
                flatten_relation_list(
                    flatten_relation_list(
                        switch_propnflattonmod(
                            sentence
                        ),
                    'flat'
                    ),
                'compound'
                ),
            'goeswith'
            )
    deprels = set(token['deprel'] for token in flat)
    for rel in deprels:
        if rel.startswith('compound') or rel.startswith('goeswith') or rel.startswith('flat'):
            flat = flatten_relation_list(flat,rel)
    return add_nulldeterminers(flat)
