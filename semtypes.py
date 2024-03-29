
class SemType:
    """
    A class representing a semantic type. It can be an AtomicType or a CompositeType,
    or a BlankType.
    
    Methods:
        is_atomic: Is this an atomic type?
        is_blank: Is this a blank type (i.e. fully wildcard)?
        is_composite: Is this a composite type?
        like(other): This is a more relaxed version of equality. Some semantic types are similar shapes but for substitution of an e for a sub-type of e, or a blank (?) for another type. This type of comparison treats those substitutions as vacuous.
        fromstring: Semantic Types can (and should) be constructed from strings consisting of letters representing basic types (e, s, t, j, i, o), blank types (?), and brackets stating how to pair these.
    """
    def __init__(self):
        self.atomic = None
        self.blank = None
        self.composite = None

    def is_atomic(self):
        return self.atomic

    def is_blank(self):
        return self.blank

    def is_composite(self):
        return self.composite

    def __repr__(self):
        return "<SemType "+str(self) + ">"

    def __str__(self):
        return 'SemType()'
    
    # Only "like" other things which have no sub-type and are just SemType.
    def like(self,other):
        return self.__class__ == other.__class__

    # Only "equal to" other things which have no sub-type and are just SemType.
    def __eq__(self,other):
        return self.__class__ == other.__class__
    
    def fromstring(string):
        if len(string) == 1:
            if string == '?':
                return BlankType()
            else:
                return AtomicType(string)
        else:
            if string.startswith("(") and string.endswith(")"):
                if string.startswith("(("): # string starts with a composite type
                    if string.endswith("))"): # string ends with a composite type
                        # now we go searching for where to split the string.
                        parencount = 0
                        for i in range(1,len(string)-2):
                            if string[i] == '(':
                                parencount += 1
                            elif string[i] == ')':
                                parencount -= 1
                            if parencount == 0:
                                return CompositeType(SemType.fromstring(string[1:i+1]),SemType.fromstring(string[i+1:-1]))
                        raise ValueError("{} is not a string corresponding to a SemType (parentheses don't pair).".format(string))
                    else: # string ends with an atomic type
                        return CompositeType(SemType.fromstring(string[1:-2]),SemType.fromstring(string[-2]))
                else: # string starts with an atomic type
                    return CompositeType(SemType.fromstring(string[1:2]),SemType.fromstring(string[2:-1]))
            else:
                raise ValueError("{} is not a string corresponding to a SemType (no parentheses surrounding).".format(string))

class AtomicType(SemType):
    """
    A sub-type of SemType, this type represents a single individual or event.
    
    Methods:
        like(other): An AtomicType is "like" another AtomicType if they are the same, if one of them is 'u' (unspecified atomic) or '?' (blank), or if one is 'e' and the other is a sub-type of 'e' ('j','i','o')
        get_content(): Returns a string representing the actual specific type this represents. 'e','s','t','j','b','i', or 'u'
    """
    basictypes = ['e','s','t','j','b','i'] # j = subject, b = object, i = indirect object
    unspecifiedatomic = 'u' # this is "like" any other atomic type
    hierarchy = {'e':['j','b','i']} # any of these lower ones are "like" type e

    def __init__(self, content):
        SemType.__init__(self)
        self.atomic = True
        self.blank = False
        self.composite = False
        if (content in AtomicType.basictypes or
                content == AtomicType.unspecifiedatomic):
            self.content = content
        else:
            raise ValueError("{} is not an appropriate atomic type.".format(content))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.get_content() == other.get_content()
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.get_content()
    
    def get_content(self):
        return self.content
    
    def like(self,other):
        if isinstance(other, self.__class__):
            if (self.get_content() == AtomicType.unspecifiedatomic or
                        other.get_content() == AtomicType.unspecifiedatomic or
                        self.get_content() == other.get_content()):
                return True
            elif (self.get_content() in AtomicType.hierarchy.keys() and other.get_content() in AtomicType.hierarchy[self.get_content()]):
                return True
            elif (other.get_content() in AtomicType.hierarchy.keys() and self.get_content() in AtomicType.hierarchy[other.get_content()]):
                return True
            else:
                return False
        else:
            return isinstance(other,SemType) and other.is_blank()
    
    def tuple_shaped_str(self):
        return self.get_content()

class CompositeType(SemType):
    """
    A sub-type of SemType, this type represents an ordered pair of two other types.
    
    Methods:
        like(other): An AtomicType is "like" another AtomicType if they are the same, if one of them is 'u' (unspecified atomic) or '?' (blank), or if one is 'e' and the other is a sub-type of 'e' ('j','i','o')
        get_left: Returns the first element of the ordered pair - another SemType.
        get_right: Returns the second element of the ordered pair - another SemType.
        uncurried_signature: If we peel apart all of the layers of functions-returning-functions-returning-functions until the right-hand side is just atomic, we get a list of types to feed into the left-hand side, then the atomic type that finally comes out the right.
        tuple_shaped_str: Alternative string representation as several inputs going to one output, like "(e,s,s) -> t".
    """
    def __init__(self,left,right):
        SemType.__init__(self)
        if not isinstance(left, SemType):
            raise ValueError("{} - passed to left entry - is not a semantic type.".format(str(left)))
        if not isinstance(right, SemType):
            raise ValueError("{} - passed to right entry - is not a semantic type.".format(str(left)))
        self.atomic = False
        self.blank = False
        self.composite = True
        self.left = left
        self.right = right
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return ((self.get_left() == other.get_left()) and (self.get_right() == other.get_right()))
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __str__(self):
        return "(" + str(self.get_left()) + str(self.get_right()) + ")"
    
    def get_left(self):
        return self.left

    def get_right(self):
        return self.right
    
    def like(self,other):
        if isinstance(other, self.__class__):
            return (self.get_left().like(other.get_left()) and
                        self.get_right().like(other.get_right()))
        else:
            return isinstance(other,SemType) and other.is_blank()
    
    def uncurried_signature(self):
        if self.get_right().is_composite():
            rightdomain, rightcodomain = self.get_right().uncurried_signature()
            return [self.get_left()] + rightdomain, rightcodomain
        else:
            return [self.get_left()], self.get_right()
    
    def tuple_shaped_str(self):
        domain, codomain = self.uncurried_signature()
        if len(domain) == 1 and not domain[0].is_composite():
            return domain[0].tuple_shaped_str() + "->" + str(codomain)
        else:
            return "(" + ",".join(x.tuple_shaped_str() for x in domain)+") -> " + str(codomain)


# this is "like" any other type but only equal to other BlankTypes.
class BlankType(SemType):
    """
    A sub-type of SemType, this type represents a wildcard - could be any type.
    
    Methods:
        like(other): An BlankType is "like" any other type.
    """

    def __init__(self):
        self.atomic = False
        self.blank = True
        self.composite = False

    def __str__(self):
        return "?"

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def like(self, other):
        return isinstance(other, SemType)
    
    def tuple_shaped_str(self):
        return "?"