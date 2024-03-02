# Overview
This repository contains code used for a project creating typed semantic compositions for Universal Dependencies syntactic trees.

Any dependencies are included in `requirements.txt`.

The main program is `pmbparsing.py`. Results in the paper were obtained by running it from the command line on 6 cpus and on PMB 4.0.0, with options `-p pmb-4.0.0 -o results -f filepairsfull.txt -n 2 -b data/mappings -m 20000`, where `data/mappings` is the same folder of data patterns used by UD-Boxer to label synsets, `pmb-4.0.0` contains the Parallel Meaning Bank data, and `filepairsfull` is a csv where the program will print pairs of one PMB reference DRS and one computed DRS, for easy cmparisn later.

Evaluation is done through Counter, using some helper functions in `counterutils.py`. Evaluation results in the paper were obtained by running this from the command line on 8 cpus and with options `-i filepairsfull.txt -n 8`.

The file `clfutils.py` provides certain functions that help one work with DRS structures in CLF format. In particular, it provides functions for converting from SBN (the output of the baseline UD-Boxer) to CLf. UD-Boxer's output was evaluated by converting to CLF using these functions and then evaluating using `counterutils.py`.

The files in `/conllus` and the file `demos.py` serve to give demonstrations of the program. See the help option for `demos.py` for more information. The files in `/conllus` demonstrate some of the semantic phenomena that the system was designed to capture.

The files `reldenotations.csv` and `postemplates.csv` contain, respectively, handwritten denotations for Universal Dependencies relations and handwritten templates for what Universal Dependencies denotations of different parts-of-speech that were use to create the results in the paper. The file `tokensubs.csv` contains some further information which is mainly useful certain stages of meaning computation.

All other files provide specific tools that help with parts of the meaning computation done by `pmbparsing.py`. That is,
* `conlluutils.py` - provides some tools for working with Universal Dependencies trees in conllu format (format from the conllu package)
* `postprocessing.py` - provides some tools that clean up final computed meanings after most of the meaning composition is done.
* `preprocessing.py` - provides some tools that clean up Universal Dependencies trees before most of the meaning composition can be done.
* `sdrt.py` - provides an extension to the NLTK DRT package to allow for Segmented Discourse Representation Theory, which we need to represent the relations in the PMB.
* `semtypes.py` - provides a class of semantic types and defines their behavior. Labeling with semantic type will help other algorithms decide in what order to compose meanings.
* `treetodrs.py` - provides the main brunt of the computation work, taking words and combining them by applying functions defined by relations.


# Attribution

The files in `/conllus` all come from [the GUM corpus](http://universal.grew.fr/?corpus=UD_English-GUM@2.10), some of them subsequently edited.

The version of Python's [NLTK DRT module](https://www.nltk.org/howto/drt.html) used here is slightly modified, which was done to prevent certain variable clashes during beta-reduction. The modified file is recorded here as `drt-MODIFIED.py` in this repository. The only change is an insertion at what once was line 976 of the original `drt.py`: lines 976-994 of `drt-MODIFIED.py` are my own doing. Note that NLTK is distributed under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

The version of Counter (Rik van Noord, Lasha Abzianidze, Hessel Haagsma, and Johan Bos. **Evaluating scoped meaning representations**. LREC 2018 [\[PDF\]](https://www.aclweb.org/anthology/L18-1267.pdf)) used here is also slightly modified. This is to facilitate integrating it into a larger Python program. The modified file is included as `counder-MODIFIED.py` in this repository. Note that Counter is distributed under the [MIT license](https://mit-license.org/).