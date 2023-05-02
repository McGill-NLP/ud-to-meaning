#import os
#os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')

from DRS_parsing.evaluation import counter
from argparse import Namespace
import argparse
import time # part of a function taken from counter
import sys # for turning off printing when running counter
import io # for turning off printing when running counter
from multiprocessing import Process, Manager, freeze_support
from queue import Empty as EmptyException
import logging
import csv # writing output
import os # manipulating paths
import clfutils

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
        logging.warning('WARNING: using -ms and -p > 1 messes up printing to screen - not recommended')
        time.sleep(5)  # so people can still read the warning

    if args.ill in ['dummy', 'spar']:
        logging.warning('WARNING: by using -ill {0}, ill-formed DRSs are replaced by a {0} DRS'.format(args.ill))
        time.sleep(3)
    elif args.ill == 'score':
        logging.warning('WARNING: ill-formed DRSs are given a score as if they were valid -- results in unofficial F-scores')
        time.sleep(3)

    if args.runs > 1 and args.prin:
        logging.warning('WARNING: we do not print specific information (-prin) for runs > 1, only final averages')
        time.sleep(5)

    if args.partial:
        raise NotImplementedError('Partial matching currently does not work')
    return args

# TODO might be more elegant to use Pools in the future.
def evaluate_clf_proc(queue,list,scorelistfile=None,logfilepfx=None,default_sense=False,default_role=False):
    if logfilepfx is not None:
        logging.basicConfig(filename=logfilepfx+".log", encoding='utf-8', level=logging.DEBUG)
    while True:
        try:
            datapoint = queue.get_nowait()
            logging.info(f"Evaluating {datapoint[0]}")
            myargs = build_counter_args(datapoint[0],
                                        datapoint[1],
                                        ill='score',
                                        restarts=datapoint[2] if len(datapoint)>2 else 20,
                                        baseline=True,
                                        default_sense=default_sense,
                                        default_role=default_role)
            try:
                ans = counter.main(myargs)
                scores = [(counter.compute_f(items[0], items[1], items[2], myargs.significant, False),items[-1]) for items in ans]
                if scorelistfile:
                    scorelist = [str(x[0][2]) for x in scores]
                    with open(os.path.join(os.path.dirname(datapoint[1]),scorelistfile), 'w') as f:
                        f.write('\n'.join(scorelist))
            except ValueError:
                logging.warning(f"Did not find output for datapoint {datapoint[0]}, using NA.")
                scores = [(('NA','NA','NA'),'NA')]
            # get the best score to write down - we're doing oracle accuracy.
            bestscore = max(scores,key=lambda x:x[0][2])
            list.append({'Datapoint':datapoint[1],'Recall':bestscore[0][0],'Precision':bestscore[0][1],'FScore':bestscore[0][2],'LongResults':bestscore[1]})
        except EmptyException:
            return

def simplify_filepairs(filepairs,synset=False,box=False,theta=False):
    if not (synset or box or theta):
        return []
    else:
        outpairs = []
        for pair in filepairs:
            pathprefix = pair[1][:-4]
            with open(pair[0]) as f:
                pmbclf = f.read().split("\n")
            with open(pair[1]) as f:
                computedclf = f.read().split("\n")
            if synset:
                pmbclf = clfutils.remove_synsets(pmbclf)
                computedclf = clfutils.remove_synsets(computedclf)
            if box:
                pmbclf = clfutils.remove_boxrels(pmbclf)
                computedclf = clfutils.remove_boxrels(computedclf)
            if theta:
                pmbclf = clfutils.remove_thetaroles(pmbclf)
                computedclf = clfutils.remove_thetaroles(computedclf)
            with open(pathprefix+"-pmb-simplified.clf","w") as f:
                nlines = f.write("\n".join(pmbclf))
            with open(pathprefix+"-simplified.clf","w") as f:
                nlines = f.write("\n".join(computedclf))
            outpairs.append([pathprefix+"-pmb-simplified.clf",pathprefix+"-simplified.clf"] + pair[2:])
        return outpairs

# Just pass it the input and output files to test, it will return all the results.
def evaluate_clfs(filepairs, nproc=8, scorelistfile=None, logfilepfx = None,remove=None):
    if logfilepfx is not None:
        logging.basicConfig(filename=logfilepfx+".log", encoding='utf-8', level=logging.DEBUG,force=True)
    if not remove:
        targetfilepairs = filepairs
        simplefilepairs = []
    else:
        simplefilepairs = simplify_filepairs(filepairs,synset = False, box = ("box" in remove),theta=False)
        logging.info("Successfully created simplified versions of files.")
        if simplefilepairs:
            targetfilepairs = simplefilepairs
        else:
            targetfilepairs = filepairs
    man = Manager()
    datainputqueue = man.Queue()
    for datapoint in targetfilepairs:
        datainputqueue.put(datapoint)
    dataoutputlist = man.list()
    default_sense = remove and "synset" in remove
    default_role = remove and "theta" in remove
    procs = [Process(target=evaluate_clf_proc, args=(datainputqueue,dataoutputlist,scorelistfile,f"{logfilepfx}-proc{i}",default_sense, default_role)) for i in range(nproc)]
    logging.info(f"Successfully made the {nproc} evaluation processes.")
    text_trap = io.StringIO()
    sys.stdout = text_trap
    for proc in procs:
        proc.start()
    logging.info("Successfully started the evaluation processes.")
    for proc in procs:
        proc.join()
        proc.terminate()
    logging.info("All evaluation processes have finished.")
    sys.stdout = sys.__stdout__
    if logfilepfx is not None:
        logging.basicConfig(force=True)
    if simplefilepairs:
        for pair in simplefilepairs:
            if os.path.exists(pair[0]):
                    os.remove(pair[0])
            if os.path.exists(pair[1]):
                    os.remove(pair[1])
    return list(dataoutputlist)

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser("Run Counter on a batch of file pairs, printing results to a detailed csv.")
    parser.add_argument("-i","--infile",action="store",required=True,dest="infile",help="the file to read file pairs for evaluation, each pair on a line and separated by tab")
    parser.add_argument("-o","--outfile",action="store",required=True,dest="outfile",help="the file to write output to in tsv format")
    parser.add_argument("-n","--nproc",action="store",default=8,dest="nproc",type=int, help="how many processes to use when running the evaluation?")
    parser.add_argument("-l","--logfilepfx",action="store",dest="logfilepfx",help="where to write the log? just the prefix as different endings may be added")
    parser.add_argument("-s","--scorelistfile",action="store",dest="scorelistfile",help="the file to write a list of F scores for every DRS in the folder where the computed DRS are")
    parser.add_argument("-r","--remove",action="store",dest="remove",type=str,help="What information to remove? Should be a string containing zero or more of the words synset, box, and theta.")
    args = parser.parse_args()
    with open(args.infile) as f:
        infileraw = f.read()
    filepairs = [x.split('\t')+[20] for x in infileraw.split('\n')]
    for x in filepairs:
        x[2]=int(x[2])
    ans = evaluate_clfs(filepairs,nproc=args.nproc,scorelistfile=args.scorelistfile,logfilepfx=args.logfilepfx,remove=args.remove)
    with open(args.outfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Datapoint','Precision','Recall','FScore','LongResults'])
        nlines = writer.writeheader()
        for row in ans:
            nlines = writer.writerow(row)

