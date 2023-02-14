# Natural Language Toolkit: Discourse Representation Theory (DRT)
#
# Author: Dan Garrette <dhgarrette@gmail.com>
#
# Copyright (C) 2001-2022 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#
# Edited by Laurestine Bradford for SDRT for PMB.

from functools import reduce

from nltk.sem.logic import (
    Expression,
    TRUTH_TYPE,
)

from nltk.sem.drt import *

class SdrtTokens(DrtTokens):
    BACKGROUND = "_BACKGROUND_"
    CONTINUATION = "_CONTINUATION_"
    CONTRAST = "_CONTRAST_"
    COMMENTARY = "_COMMENTARY_"
    ELABORATION = "_ELABORATION_"
    EXPLANATION = "_EXPLANATION_"
    INSTANCE = "_INSTANCE_"
    NARRATION = "_NARRATION_"
    NECESSITY = "_NECESSITY_"
    PARALLEL = "_PARALLEL_"
    POSSIBILITY = "_POSSIBILITY_"
    PRESUPPOSITION = "_PRESUPPOSITION_"
    PRECONDITION = "_PRECONDITION_"
    RESULT = "_RESULT_"
    TOPIC = "_TOPIC_"

    BOXRELS = [BACKGROUND,
        CONTINUATION,
        CONTRAST,
        COMMENTARY,
        ELABORATION,
        EXPLANATION,
        INSTANCE,
        NARRATION,
        NECESSITY,
        PARALLEL,
        POSSIBILITY,
        PRESUPPOSITION,
        PRECONDITION,
        RESULT,
        TOPIC]

    TOKENS = DrtTokens.TOKENS + BOXRELS


class SdrtParser(DrtParser):
    """A lambda calculus expression parser."""

    def __init__(self):
        DrtParser.__init__(self)
        self.operator_precedence = self.operator_precedence | dict(
                                        [(x, 2) for x in SdrtTokens.BOXRELS])

    def get_all_symbols(self):
        """This method exists to be overridden"""
        return SdrtTokens.SYMBOLS

    def handle(self, tok, context):
        """This method is intended to be overridden for logics that
        use different operators or expressions"""
        if tok in DrtTokens.NOT_LIST:
            return self.handle_negation(tok, context)
        
        elif tok in SdrtTokens.BOXRELS:
            return self.handle_boxrel(tok, context)

        elif tok in DrtTokens.LAMBDA_LIST:
            return self.handle_lambda(tok, context)

        elif tok == DrtTokens.OPEN:
            if self.inRange(0) and self.token(0) == DrtTokens.OPEN_BRACKET:
                return self.handle_DRS(tok, context)
            else:
                return self.handle_open(tok, context)

        elif tok.upper() == DrtTokens.DRS:
            self.assertNextToken(DrtTokens.OPEN)
            return self.handle_DRS(tok, context)

        elif self.isvariable(tok):
            if self.inRange(0) and self.token(0) == DrtTokens.COLON:
                return self.handle_prop(tok, context)
            else:
                return self.handle_variable(tok, context)

    def make_BoxRelationExpression(self, relation, drs):
        return SdrtBoxRelationExpression(relation, drs)

    def handle_boxrel(self,tok,context):
        return self.make_BoxRelationExpression(tok,self.process_next_expression(tok))

class SdrtExpression(DrtExpression):
    _drt_parser = SdrtParser()

class SdrtBoxRelationExpression(SdrtExpression):
    def __init__(self, relation, drs):
        self.relation = relation
        self.drs = drs

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.relation == other.relation
            and self.drs == other.drs
        )

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__

    def fol(self):
        return self.drs.fol()

    def __str__(self):
        return f"{self.relation}({self.drs})"

    def eliminate_equality(self):
        return SdrtBoxRelationExpression(self.relation, self.drs.eliminate_equality())

    def free(self):
        return self.drs.free()

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
            return SdrtBoxRelationExpression(
                self.relation,
                self.drs.replace(variable, expression, replace_bound, alpha_convert),
            )

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.drs)])
    
    def simplify(self):
        return SdrtBoxRelationExpression(self.relation,self.drs.simplify())

    @property
    def type(self):
        return TRUTH_TYPE

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        return self.drs.get_refs(recursive)

    def _pretty(self):
        drs_s = self.drs._pretty()
        blank = " " * len("%s" % self.relation)
        return (
            [blank + " " + line for line in drs_s[:1]]
            + ["%s" % self.relation + " " + line for line in drs_s[1:2]]
            + [blank + " " + line for line in drs_s[2:]]
        )

    def negate(self):
        """:see: Expression.negate()"""
        return DrtNegatedExpression(self.drs)
