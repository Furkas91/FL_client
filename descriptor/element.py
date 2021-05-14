from descriptor import Descriptor


class Element(Descriptor):

    def read(self):
        return self._value


class BoolElement(Element):
    pass


class IntElement(Element):
    pass


class StrElement(Element):
    pass


class FloatElement(Element):
    pass


class ByteElement(Element):
    pass

