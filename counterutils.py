import os
os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')

from DRS_parsing.evaluation import counter
from argparse import Namespace # to make arguments for counter
import time # part of a function taken from counter
import sys # for turning off printing when running counter
import io # for turning off printing when running counter

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

# Just pass it the input and output files to test, it will return all the results.
def evaluate_clfs(filepairs):
    evalresults = []
    # Evaluate each datapoint in turn.
    for datapoint in filepairs:
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
            evalresults.append({'Datapoint':datapoint[1],'Recall':bestscore[0],'Precision':bestscore[1],'FScore':bestscore[2]})
        except Exception as e:
            print(e)
            evalresults.append({'Datapoint':datapoint[1],'Precision':'NA','Recall':'NA','FScore':'NA'})
    # TODO multiprocessing goes here
    return evalresults

if __name__ == "__main__":
    # TODO this command line tool
    # when called from the command line it needs an input file with pairs of filenames,
    # also a results file to write to
    print("I promise I'll implement this :)")
