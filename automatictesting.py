import os
os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')

from DRS_parsing.evaluation import counter
import stanza # parsing to UD
from simpleparsing import *
from semtypes import *
from argparse import Namespace # to make arguments for counter
import time # part of a function taken from counter
import sys # for turning off printing when running counter
import io # for turning off printing when running counter
import csv # writing csvs

# This part sets up the NLP pipeline we will need to get UD parses.
stanzanlp = stanza.Pipeline(lang='en',
                processors = 'tokenize,pos,lemma,depparse'
)
#                tokenize_pretokenized=True)

# Helps in converting NLTK DRS data structures to lists of clauses for CLF format.
# It takes as input a condition from the DRT module, the name of the projective DRS it belongs to,
# and a counter saying what names are still available for future DRS boxes.
# It returns a list of clauses and an updated counter.
def process_cond(cond, mydrsname, counter):
    if isinstance(cond,DrtApplicationExpression):
        function, args = cond.uncurry()
        return([f'{mydrsname} {function} ' + ' '.join(str(x) for x in args)], counter)
    elif isinstance(cond,DrtProposition):
        var = cond.variable
        embdrs = cond.drs
        embname = 'b'+str(counter)
        newline = f'{mydrsname} PRP {var} {embname}'
        newlines, counter = to_clf(embdrs,counter)
        return [newline] + newlines, counter
    elif isinstance(cond,DRS):
        return to_clf(cond,counter)
    elif isinstance(cond,DrtNegatedExpression):
        embterm = cond.term
        embname = 'b'.str(counter)
        newline = f'{mydrsname} NOT {embname}'
        if not isinstance(embterm,DRS):
            counter = counter + 1
        newlines, counter = process_cond(embterm,embname,counter)
        return [newline] + newlines, counter
    elif isinstance(cond,DrtOrExpression):
        embterm1 = cond.first
        embname1 = 'b'+str(counter)
        if not isinstance(embterm1,DRS):
            counter = counter + 1
        newlines1, counter = process_cond(embterm1,embname1,counter)
        embterm2 = cond.second
        embname2 = 'b'+str(counter)
        if not isinstance(embterm2,DRS):
            counter = counter + 1
        newlines2, counter = process_cond(embterm2,embname2,counter)
        newline = f'{mydrsname} DIS {embname1} {embname2}'
        return [newline] + newlines1 + newlines2, counter
    elif isinstance(cond,DrtConcatenation):
        embterm1 = cond.first
        embname1 = 'b'+str(counter)
        if not isinstance(embterm1,DRS):
            counter = counter + 1
        newlines1, counter = process_cond(embterm1,embname1,counter)
        embterm2 = cond.second
        if not isinstance(embterm2,DRS):
            counter = counter + 1
        newlines2, counter = process_cond(embterm2,embname2,counter)
        newline = f'{mydrsname} CNJ {embname1} {embname2}'
        return [newline] + newlines1 + newlines2, counter
    elif isinstance(cond,DrtEqualityExpression):
        return [f'{mydrsname} EQU {cond.first} {cond.second}'], counter
    elif isinstance(cond,DrtBinaryExpression):
        embterm1 = cond.first
        embname1 = 'b'+str(counter)
        if not isinstance(embterm1,DRS):
            counter = counter + 1
        newlines1, counter = process_cond(embterm1,embname1,counter)
        embterm2 = cond.second
        if not isinstance(embterm2,DRS):
            counter = counter + 1
        newlines2, counter = process_cond(embterm2,embname2,counter)
        newline = f'{mydrsname} {cond.getOp()} {embname1} {embname2}'
        return  [newline] + newlines1 + newlines2, counter

# Converting NLTK DRS data structures to lists of clauses for CLF format.
# It takes as input a DRS from the DRT module,
# and a counter saying what names are still available for future DRS boxes.
# It returns a list of clauses.
# If it is being recursively called (or is called with a high counter number)
# it also returns an updated counter
def to_clf(drs, counter=0):
    istop = counter==0
    mydrsname = 'b'+str(counter)
    counter = counter + 1
    clflines = []
    for x in drs.refs:
        clflines.append(mydrsname + " REF " + str(x))
    for cond in drs.conds:
        newlines, counter = process_cond(cond,mydrsname,counter)
        clflines = clflines + newlines
    if istop:
        return clflines
    return clflines, counter

# This simply helps to pass arguments to Counter, since Counter expects command line arguments.
# Most of the code in this function is taken from the Counter code itself and is not mine.
def build_counter_args(f1, f2,
                    restarts=20,
                    parallel=1,
                    mem_limit=1000,
                    smart='conc',
                    prin=False,
                    ms=False,
                    ms_file='',
                    all_idv=False,
                    significant=4,
                    stats='',
                    detailed_stats=0,
                    no_mapping=False,
                    sig_file='',
                    codalab='',
                    ill='error',
                    runs=1,
                    max_clauses=0,
                    baseline=False,
                    partial=False,
                    default_sense=True,
                    default_role=False,
                    default_concept=False,
                    include_ref=False):
    args = Namespace(f1=f1,
                        f2=f2,
                        restarts=restarts,
                        parallel=parallel,
                        mem_limit=mem_limit,
                        smart=smart,
                        prin=prin,
                        ms=ms,
                        ms_file=ms_file,
                        all_idv=all_idv,
                        significant=significant,
                        stats=stats,
                        detailed_stats=detailed_stats,
                        no_mapping=no_mapping,
                        sig_file=sig_file,
                        codalab=codalab,
                        ill=ill,
                        runs=runs,
                        max_clauses=max_clauses,
                        baseline=baseline,
                        partial=partial,
                        default_sense=default_sense,
                        default_role=default_role,
                        default_concept=default_concept,
                        include_ref=include_ref)
    # Check if combination of arguments is valid
    if args.ms and args.runs > 1:
        raise NotImplementedError("Not implemented to average over individual scores, only use -ms when doing a single run")
    if args.restarts < 1:
        raise ValueError('Number of restarts must be larger than 0')

    if args.ms and args.parallel > 1:
        print('WARNING: using -ms and -p > 1 messes up printing to screen - not recommended')
        time.sleep(5)  # so people can still read the warning

    if args.ill in ['dummy', 'spar']:
        print('WARNING: by using -ill {0}, ill-formed DRSs are replaced by a {0} DRS'.format(args.ill))
        time.sleep(3)
    elif args.ill == 'score':
        print ('WARNING: ill-formed DRSs are given a score as if they were valid -- results in unofficial F-scores')
        time.sleep(3)

    if args.runs > 1 and args.prin:
        print('WARNING: we do not print specific information (-prin) for runs > 1, only final averages')
        time.sleep(5)

    if args.partial:
        raise NotImplementedError('Partial matching currently does not work')
    return args

# This function takes a list of lines in CLF format
# and simplifies them by removing the information I can't get from them:
# extra information that comes with names,
# tense information,
# and theta roles.
def simplify_clf(clflines):
    clflinespctsplit = [x.split("%") for x in clflines]
    textwords = set(x[1] for x in clflinespctsplit if len(x)>1)
    # Remove lowercase lines that come from the same word as a Name line.
    if textwords:
        linestodrop = []
        for word in textwords:
            sameword = [x for x in clflinespctsplit if (len(x)>1 and x[1] == word)]
            if sameword and max((x[0][3:7] == "Name") for x in sameword): # if this came from the same word
                for x in sameword:
                    if len(x[0]) > 3 and x[0][3].islower():
                        linestodrop.append(x)
        clflinespctsplit = [x for x in clflinespctsplit if x not in linestodrop and len(x) > 0]
    clflinestokens = [(x[0].split(),x[1:]) for x in clflinespctsplit]
    # Remove the first argument on the line if it's a "sense" disambiguation thing
    for i in range(len(clflinestokens)):
        x, y = clflinestokens[i]
        if len(x) > 2:
            if len(x[2]) > 2 and x[2][0]=='"' and x[2][1].isalpha() and x[2][2] == '.' and x[2][3:-1].isdigit() and x[2][-1] == '"':
                clflinestokens[i] = (x[:2] + x[3:],y)
    # Change any ArgN or theta-roles to just Arg.
    for x,_ in clflinestokens:
        if len(x) > 1 and ((x[1] in ("Agent","Theme","Topic","Recipient","Experiencer")) or (
                x[1].startswith('Arg') and x[1][3:].isdigit())):
            x[1] = "Arg"
    # Remove lines that include variable t if t participates in a Time relation as the second argument.
    timevars = [x[-1] for x,y in clflinestokens if len(x)>1 and x[1]=='Time']
    for t in timevars:
        clflinestokens = [(x,y) for x,y in clflinestokens if t not in x[2:]]
    # TODO Flatten all PRESUPPOSITION relations
    return(['%'.join([' '.join(line[0])+' '] + line[1]) for line in clflinestokens])

# Loops through all DRS files in the passed folder, and writes output files in the passed outdir.
# Also returns a list of pairs of one PMB file and one DRS file.
def pmb_files_to_meaning(pmbdir, outdir):
    pmbfiles = []
    for path, _, files in os.walk(pmbdir):
        pmbfiles = pmbfiles + [path + '\\' + x for x in files]
    datapointprefixes = [".".join(x.split(".")[:-2]) for x in pmbfiles]
    datapointpathdict = dict((x,{'drs':x+'.drs.clf','tokens':x+'.tok.off','raw':x+'.raw'}) for x in datapointprefixes if x+'.drs.clf' in pmbfiles and x+'.tok.off' in pmbfiles and x+'.raw' in pmbfiles)
    
    # Make the output directory.
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    returnlist = []
    
    i = 0
    # We'll process each datapoint in turn.
    for dpname in datapointpathdict.keys():
        print(dpname)
        # There are certain individual PMB files that give it trouble; these are currently tracked here.
        if dpname in ("..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p04\\d2024\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p14\\d3314\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p19\\d1455\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p23\\d2513\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p39\\d2847\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p49\\d1795\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p50\\d0704\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p59\\d3229\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p74\\d0808\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p75\\d3043\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p80\\d0554\\en"):
            continue
        # This bit of code lets you set certain indices to skip over
        if i < 10:
            i+=1
        else:
            i+=1
            continue
        try:
            # read the token file
            with open(datapointpathdict[dpname]['tokens']) as f:
                tokensraw = f.read()
            with open(datapointpathdict[dpname]['raw']) as f:
                textraw = f.read()
            tokens = ['~'.join(x.split(' ')[3:]) for x in tokensraw.split('\n') if len(x) > 0]
            ud_dict = stanzanlp(textraw).to_dict()[0]
            for x in ud_dict:
                x['form'] = x['text']
            if not os.path.exists(outdir+'\\'+dpname.replace('\\','-')):
                os.mkdir(outdir+'\\'+dpname.replace('\\','-'))
            ud_parse = conllu.TokenList(ud_dict)
            # UD parse to an output file, for testing
            ud_out = ud_parse.serialize()
            with open(myoutdir+'\\'+dpname.replace('\\','-')+'\\'+'udparse.conll',mode='w') as f:
                nlines = f.write(ud_out)
            # DRSs to an output file
            ud_drss = getalldens(tokenlist=ud_parse)
            clflines = [to_clf(den) for semtype,den in ud_drss if isinstance(den,DRS)]
            with open(outdir+'\\'+dpname.replace('\\','-')+'\\'+'drsoutput.clf', 'w') as f:
                nlines = f.write('\n\n'.join(['\n'.join(x) for x in clflines]))
            # change both of them to simplified files
            with open(outdir+'\\'+dpname.replace('\\','-')+'\\'+'drsoutputsimple.clf', 'w') as f:
                nlines = f.write('\n\n'.join(['\n'.join(simplify_clf(x)) for x in clflines]))
            pmbclf = ''
            with open(datapointpathdict[dpname]['drs']) as f:
                pmbclf = f.read()
            pmbclflines = pmbclf.split('\n')
            simplifiedpmbname = '.'.join(datapointpathdict[dpname]['drs'].split('.')[:-2] + [datapointpathdict[dpname]['drs'].split('.')[-1]+"simple.clf"])
            with open(simplifiedpmbname, 'w') as f:
                nlines = f.write('\n'.join(simplify_clf(pmbclflines)))
            returnlist.append((simplifiedpmbname,outdir+'\\'+dpname.replace('\\','-')+'\\'+'drsoutputsimple.clf'))
        except Exception as e:
            continue

    return returnlist

def score_computed_drss(pmbdrslist,resultsfile):
    csvrows = []
    # Evaluate each datapoint in turn.
    for datapoint in pmbdrslist:
        print(datapoint[1])
        try:
            text_trap = io.StringIO()
            sys.stdout = text_trap
            myargs = build_counter_args(datapoint[0],
                                        datapoint[1],
                                        ill='score',
                                        baseline=True)
            try:
                ans = counter.main(myargs)
                scores = [counter.compute_f(items[0], items[1], items[2], myargs.significant, False) for items in ans]
            except ValueError:
                scores = [('NA','NA','NA')]
            sys.stdout = sys.__stdout__
            counteroutput = text_trap.getvalue()
            # get the best score to write down - we're doing oracle accuracy.
            bestscore = max(scores,key=lambda x:x[2])
            csvrows.append({'Datapoint':datapoint[1],'Recall':bestscore[0],'Precision':bestscore[1],'FScore':bestscore[2]})
        except Exception as e:
            print(e)
            csvrows.append({'Datapoint':datapoint[1],'Precision':'NA','Recall':'NA','FScore':'NA'})

    # write csv
    with open(resultsfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Datapoint','Precision','Recall','FScore'])
        nlines = writer.writeheader()
        for row in csvrows:
            nlines = writer.writerow(row)

    return csvrows

pmbdir = "pmb-gold-english"
myoutdir = "results"
resultsfile = "results\\latestresults.csv" # This path is computed from the working directory not the myoutdir

drspairs = pmb_files_to_meaning(pmbdir, myoutdir)

csvrows = score_computed_drss(drspairs,resultsfile)