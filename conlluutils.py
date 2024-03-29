import conllu # reading ConLL-U files
import copy # deep-copying Tokens and and TokenLists

def stanza_to_conllu(doc):
    ud_dict = doc.to_dict()[0]
    for x in ud_dict:
        x['form'] = x['text']
        if 'feats' in x.keys() and x['feats']:
            x['feats'] = dict((y.split('=')[0],y.split('=')[1]) for y in x['feats'].split('|'))
        else:
            x['feats'] = None
    try:
        ud_parse = conllu.TokenList(ud_dict)
        tree = ud_parse.to_tree()
    # In case the TokenList is ill-formed.
    except conllu.exceptions.ParseException:
        return conllu.TokenList([dict(id=1,form='event',lemma='event',upos='VERB',xpos='',feats={},head=0,deprel='root')])
    return ud_parse

# This function takes a TokenTree as input,
# and returns a new TokenTree with most data the same as the original TokenTree,
# with two differences:
# first, the new TokenTree now has no children,
# and second, the "form" and "lemma" of the TokenTree's token are now
# concatenations of the existing "form" and "lemma" with those of all children
# and their descendants as well.
def concat_with_all_dependents(treenode):
    '''
    This function takes a TokenTree as input,
    and returns a new TokenTree with most data the same as the original TokenTree,
    with two differences:
    first, the new TokenTree now has no children,
    and second, the "form" and "lemma" of the TokenTree's token are now
    concatenations of the existing "form" and "lemma" with those of all children
    and their descendants as well.

            Parameters:
                    treenode: A TokenTree (from conllu package)

            Returns:
                    flattened: A new TokenTree with all of the children concatenated as part of the one root.
    '''
    treenode = copy.deepcopy(treenode)
    for child in treenode.children:
        child = concat_with_all_dependents(child)
        if child.token['id'] < treenode.token['id']:
            treenode.token['form'] = child.token['form'] + '_' + treenode.token['form']
            treenode.token['lemma'] = child.token['lemma'] + '_' + treenode.token['lemma']
        else:
            treenode.token['form'] = treenode.token['form'] + '_' + child.token['form']
            treenode.token['lemma'] = treenode.token['lemma'] + '_' + child.token['lemma']
    treenode.children = []
    return treenode

def tree_to_tokenlist(treenode):
    '''
    This function is the reverse of a function from conllu that turns a TokenList into a tree.
    It turns a TokenTree back into a TokenList.
    NOTE: It uses the same actual tokens, not copies!

            Parameters:
                    sentence: A TokenTree (from conllu package).

            Returns:
                    result: A TokenList of all the tokens in TokenTree, sorted by index.
    '''
    if len(treenode.children) == 0:
        return [treenode.token]
    else:
        answer = [treenode.token]
        for child in treenode.children:
            answer = answer + tree_to_tokenlist(child)
        return conllu.TokenList(sorted(answer, key = lambda x:x['id']))

def reindex_tokenlist(sentence):
    '''
    This function takes a TokenList as input,
    and returns a new TokenList,
    mostly the same as the input,
    but with new id's starting at 1 and increasing by 1 with each subsequent word,
    and all heads updated to match the new id's.
    Handy when you've been adding tokens so the id's aren't all nice and normal yet.

            Parameters:
                    sentence: A TokenList (from conllu package).

            Returns:
                    result: A new TokenList with the indices increasing one by one, otherwise same as input.
    '''
    sentence = copy.deepcopy(sentence)
    i = 0
    for token in sentence:
        i += 1
        if 'misc' not in token.keys() or not token['misc']:
            token['misc'] = {}
        token['misc'].update(oldid = token['id'])
        token['id'] = i
    for token in sentence:
        newhead = 0
        for token2 in sentence:
            if token2['misc']['oldid'] == token['head']:
                newhead = token2['id']
        token['head'] = newhead
    for token in sentence:
        del token['misc']['oldid']
    return sentence

def flatten_relation_tree(treenode,relation,sep="_"):
    '''
    This function flattens a given relation.
    That is, it takes a TokenTree
    and a relation (a string corresponding to a value of the "deprel" field) as input.
    It returns a new TokenTree, just like the input, except
    anywhere there used to be that relation,
    - the dependent is completely flattened, with all _its_ dependents collapsed onto it in one token,
    - and then the dependent is concatenated to the head as part of that token, and removed.
    The result is that we essentially treat that relation as being between two parts of the same word,
    and combine the parts of that word,
    with POS and syntactic role corresponding to the relation's head.

            Parameters:
                    sentence: A TokenList (from conllu package)
                    relation: A string representing the relation to flatten
                    sep: The string to use as a separator when concatenating dependents of that relation.

            Returns:
                    result: A new TokenTree with all of those relations removed and their children concatenated.
    '''
    treenode = copy.deepcopy(treenode)
    childrentokeep = []
    for child in treenode.children:
        if child.token['deprel'] == relation:
            child = concat_with_all_dependents(child)
            if child.token['id'] < treenode.token['id']:
                treenode.token['form'] = child.token['form'] + sep + treenode.token['form']
                treenode.token['lemma'] = child.token['lemma'] + sep + treenode.token['lemma']
            else:
                treenode.token['form'] = treenode.token['form'] + sep + child.token['form']
                treenode.token['lemma'] = treenode.token['lemma'] + sep + child.token['lemma']
        else:
            child = flatten_relation_tree(child, relation, sep)
            childrentokeep.append(child)
    treenode.children = childrentokeep
    return treenode

def flatten_relation_list(sentence,relation,sep="_"):
    '''
    This function takes a TokenList as input
    and performs the same operation as flatten_relation_tree on it,
    returning the resulting TokenList with new indices.

            Parameters:
                    sentence: A TokenList (from conllu package)
                    relation: A string representing the relation to flatten
                    sep: The string to use as a separator when concatenating dependents of that relation.

            Returns:
                    result: A new TokenList with all of those relations removed and their children concatenated.
    '''
    return conllu.TokenList(reindex_tokenlist(
                tree_to_tokenlist(
                    flatten_relation_tree(
                        sentence.to_tree(), relation, sep
                    ))))

