# Demos with a few Conll files that I have hand-selected.
import os # changing working directory
os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
# the modules we need should be in the working directory now, and the conllu files.
from semtypes import *
from simpleparsing import *

# basic transitive sentence
# We read the file, and print its structure...
with open("conllus\\the rider possesses the will.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# adjectives and binarization
with open("conllus\\the city enjoys a temperate climate.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# quantifiers
with open("conllus\\each section has a theme.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# obliques!
with open("conllus\\the first drops fell onto the parched stones from a cloudless blue sky.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# nominal modifiers
with open("conllus\\after a brief period in frankfurt the family moved to basel.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# determinerless nouns!
with open("conllus\\the first drops of rain fell onto the parched stones from a cloudless blue sky.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# determinerless nouns!
with open("conllus\\pages hung in tatters from the sodden blue spine.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# determinerless nouns!
with open("conllus\\successful stormtroopers share a high - five.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# determinerless nouns and numbers!
with open("conllus\\smith offered three reasons.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# proper nouns as det+noun, allowing them to be modified!
with open("conllus\\after a brief period in frankfurt the family moved to basel in switzerland.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# flat constructions!
with open("conllus\\wikinews interviews meteorological experts on cyclone phalin.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# flat and compound constructions, with special case for proper nouns
with open("conllus\\texas student ahmed mohamed inspires social movement.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# ccomp relation
# note that the answer does not simplify because of the band-aid on the DRT module
with open("conllus\\the authors say the results confirm the existence of inadequate iodine intake in the australian population.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# csubj and ccomp
with open("conllus\\ending inflation means freeing all americans from the terror of runaway living costs.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# coordinate structures with nouns
with open("conllus\\most iodine in food comes from seafood milk and salt.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()


# The coordination doesn't work anymore with the new denotations. This is something to fix.
# coordinate structures with nouns and adjectives
with open("conllus\\fraud and corruption prevent a fair and proper expression of the public voice.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()

# coordinate structures with verb phrases
with open("conllus\\during the middle ages athens experienced a decline but re-emerged under byzantian rule.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()

# acl relation - relative clauses
with open("conllus\\another contributory factor has been the decreasing consumption of iodized salt in foods.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()

# acl and advcl
with open("conllus\\through this analysis a categorization of themes emerges iillustrating tactics for negotiating intertexts and paratexts unique to each group of fans.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()

# acl relation - clauses as complements of nouns, with advcl also
with open("conllus\\if this were the case it would support the idea that non-avian theropods have relatively thinner femora.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()

# xcomp relation
with open("conllus\\id like to go out to dinner though.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()

# xcomp relation with object-to-subject control
with open("conllus\\ill have melanie call you.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()

# advcl relation
with open("conllus\\therefore both institutes joined forces to develop a set of clinics on dh for librarians.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()

# advcl relation
with open("conllus\\youre so stupid thinking I spent the night.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()

# advcl relation and acl in a long sentence
# note: this one is very slow.
with open("conllus\\it is the oath created by senators when they tried the first impeachment of a president in 1868.conll") as f:
    testconllu = f.read()
somedens = getalldens(rawconllu=testconllu,withtrace=True)
somedens[0][1].pretty_print()
#tracetogvtree(somedens[0]).view()

