import re # need this to get SemTypes from strings

class SemType:
    def __init__(self):
        self.atomic = None
        self.blank = None

    def is_atomic(self):
        return self.atomic

    def is_blank(self):
        return self.blank

    def __repr__(self):
        return "<SemType "+str(self) + ">"
    
    # TODO this needs to be fixed -
    # it doesn't always choose a bracketing that will succeed...
    def fromstring(string):
        if len(string) == 1:
            if string == '?':
                return BlankType()
            else:
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
    unspecifiedatomic = 'u' # this is "like" any other atomic type

    def __init__(self, content):
        SemType.__init__(self)
        self.atomic = True
        self.blank = False
        if content in AtomicType.basictypes or content == AtomicType.unspecifiedatomic:
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
            return (self.get_content() == AtomicType.unspecifiedatomic or
                        other.get_content() == AtomicType.unspecifiedatomic or
                        self.get_content() == other.get_content()
                        )
        else:
            return isinstance(other,SemType) and other.is_blank()

class CompositeType(SemType):
    def __init__(self,left,right):
        SemType.__init__(self)
        if not isinstance(left, SemType):
            raise ValueError("{} - passed to left entry - is not a semantic type.".format(str(left)))
        if not isinstance(right, SemType):
            raise ValueError("{} - passed to right entry - is not a semantic type.".format(str(left)))
        self.atomic = False
        self.blank = False
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

# this is "like" any other type but only equal to other BlankTypes.
class BlankType(SemType):
    def __init__(self):
        self.atomic = False
        self.blank = True

    def __str__(self):
        return "?"

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def like(self, other):
        return isinstance(other, SemType)
