import conllu # reading ConLL-U files
import copy # deep-copying Tokens and and TokenLists
from conlluutils import *

tokensubs = []
with open("tokensubs.csv") as f:
    tokensubs = [x.split() for x in f.read().split('\n') if '\t' in x]

def switch_propnflattonmod(sentence):
    '''
    This function is a hack,
    but it changes any "flat" dependency relations
    with head a NOUN and dependent a PROPN
    into "nmod"...
    or into "flat" dependent on a previous PROPN
    if there are multiple on the same NOUN
    ... it just makes the denotations come out nicer.
    It takes a TokenList as input,
    and returns a new TokenList with the change.
    
            Parameters:
                    sentence: A TokenList representing the sentence to preprocess.
            Returns:
                    postprocessed: A TokenList representing the modified sentence.
    '''
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

def add_nulldeterminers(sentence):
    '''
    Takes a TokenList as input and returns a new TokenList, very similar to the first,
    but with null definite determiners inserted just before proper nouns without existing determiners,
    and null indefinite determiners inserted just before common nouns without existing determiners.
    Because of a hack involving the "flat" relation,
    no determiner is inserted if the noun is related to its head with the "amod" relation.
    
            Parameters:
                    sentence: A TokenList representing the sentence to preprocess.
            Returns:
                    postprocessed: A TokenList representing the modified sentence.
    '''
    determinered = [sentence.filter(id=token['head'])[0] for token in sentence
                        if (token['upos']=='DET' and token['deprel'].startswith('det'))]
    determinerless = [token for token in sentence
                        if (token['upos'] in ('NOUN','PROPN','PRON')
                            and token not in determinered
                            and token['deprel'] != "amod")]
    for token in determinerless:
        sentence.append(conllu.models.Token({'id':token['id']-0.5,
                            'form':'0def' if token['upos'] in ('PROPN','PRON') else '0indf',
                            'lemma':'0def' if token['upos'] in ('PROPN','PRON') else '0indf',
                            'upos':'DET-DEF' if token['upos'] in ('PROPN','PRON') else 'DET-INDF',
                            'xpos':None,
                            'feats':None,
                            'head':token['id'],
                            'deprel':'det',
                            'deps': None,
                            'misc': None
                        }))
    sentence = conllu.TokenList(sorted(sentence,key=lambda x:x['id']))
    return reindex_tokenlist(sentence)

def clean_lemmas(sentence):
    '''
    This takes a sentence and removes characters from the lemmas and features
    that will not play well with later derivation.
    It is meant to undone by a step in postprocessing.
    
            Parameters:
                    sentence: A TokenList representing the sentence to preprocess.
            Returns:
                    postprocessed: A TokenList representing the modified sentence.
    '''
    sentence = copy.deepcopy(sentence)
    for token in sentence:
        token['lemma'] = token['lemma'].lower()
        for sub in tokensubs:
            token['lemma'] = token['lemma'].replace(sub[0],sub[1])
        token['lemma'] = token['lemma'] + ("" if len(token['lemma'])>1 else "_LETTER")
        if token['feats'] is not None:
            keypairs = [[key,token['feats'][key]] for key in token['feats'].keys()]
            for pair in keypairs:
                for sub in tokensubs:
                    pair[0] = pair[0].replace(sub[0],sub[1])
                    pair[1] = pair[1].replace(sub[0],sub[1])
            token['feats'] = dict(keypairs)
                
    return conllu.TokenList(sentence)

# Remove multi-token tokens
def remove_multitokens(sentence):
    '''
    Removes tokens that span multiple rows in the conllu.

            Parameters:
                    sentence: A TokenList representing the sentence to preprocess.
            Returns:
                    postprocessed: A TokenList representing the modified sentence.
    '''
    sentence = [x for x in sentence if isinstance(x['id'],int)]
    sentence = conllu.TokenList(sentence)
    return reindex_tokenlist(sentence)

def preprocess(sentence):
    '''
    A combined function for all of the preprocessing steps.
    it changes flat relations between Nouns and Proper Nouns to "amod" relations (for semantic reasons),
    flattens remaining "flat" relations,
    flattens remaining "compound" relations,
    and adds null determiners to any determiner-less nouns and proper nouns (except those modified by "amod").

            Parameters:
                    sentence: A TokenList representing the sentence to preprocess.
            Returns:
                    postprocessed: A TokenList representing the modified sentence.
    '''
    flat = flatten_relation_list(
                flatten_relation_list(
                    flatten_relation_list(
                        switch_propnflattonmod(
                            remove_multitokens(
                                sentence
                            )
                        ),
                    'flat', "~"
                    ),
                'compound', "_"
                ),
            'goeswith', "_"
            )
    deprels = set(token['deprel'] for token in flat)
    for rel in deprels:
        if rel.startswith('flat'):
            flat = flatten_relation_list(flat,rel,"~")
        elif  rel.startswith('compound') or rel.startswith('goeswith'):
            flat = flatten_relation_list(flat,rel,"_")
    return add_nulldeterminers(clean_lemmas(flat))
