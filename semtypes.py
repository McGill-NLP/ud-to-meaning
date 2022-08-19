import re # need this to get SemTypes from strings

class SemType:
    def __init__(self):
        self.atomic = None

    def is_atomic(self):
        return self.atomic

    def __repr__(self):
        return "<SemType "+str(self) + ">"
    
    # TODO this needs to be fixed -
    # it doesn't always choose a bracketing that will succeed...
    def fromstring(string):
        if len(string) == 1:
            return AtomicType(string)
        else:
            stringparse = re.search(r'^\(((.)|(\(.*\)))((.)|(\(.*\)))\)$',string)
            if stringparse:
                leftstring, rightstring = stringparse.group(1,4)
                return CompositeType(SemType.fromstring(leftstring),SemType.fromstring(rightstring))
            else:
                raise ValueError("{} is not a string corresponding to a SemType.".format(string))

class AtomicType(SemType):
    basictypes = ['e','s','t']

    def __init__(self, content):
        SemType.__init__(self)
        self.atomic = True
        if content in AtomicType.basictypes:
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

class CompositeType(SemType):
    def __init__(self,left,right):
        SemType.__init__(self)
        if not isinstance(left, SemType):
            raise ValueError("{} - passed to left entry - is not a semantic type.".format(str(left)))
        if not isinstance(right, SemType):
            raise ValueError("{} - passed to right entry - is not a semantic type.".format(str(left)))
        self.atomic = False
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
