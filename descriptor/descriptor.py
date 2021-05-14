from abc import ABC, abstractmethod
import re

from element import *


class Descriptor(ABC):
    def __init__(self, value):
        self.class_name = type(value)
        self._value = value

    def __str__(self):
        return str(self._value)

    def __hash__(self):
        return hash(self._value)

    @abstractmethod
    def read(self):
        pass

    @staticmethod
    def describe(value):
        types = {
            int: IntElement,
            bool: BoolElement,
            str: StrElement,
            bytes: ByteElement,
            float: FloatElement,
            list: ListDescriptor,
            dict: DictDescriptor,
            object: ObjectDescriptor,
            Descriptor: lambda x: x
        }
        try:
            return types[type(value)](value)
        except KeyError:
            if isinstance(value, Descriptor):
                return value
            else:
                return ObjectDescriptor(value)


class ListDescriptor(Descriptor):
    def __init__(self, value: list):
        super().__init__(value)
        self._value = [Descriptor.describe(elem) for elem in value]

    def read(self):
        return [elem.read() for elem in self._value]


class DictDescriptor(Descriptor):
    def __init__(self, value: dict):
        super().__init__(value)
        # Следующая страшная строчка берет создает словарь из словаря, применив ко всем его элементам describe
        self._value = dict(zip(map(Descriptor.describe, value.keys()),
                               map(Descriptor.describe, value.values())))

    def read(self):
        return dict(zip([elem.read() for elem in self._value.keys()],
                        [elem.read() for elem in self._value.values()]))


class ObjectDescriptor(Descriptor):
    def __init__(self, value: object):
        super().__init__(value)
        fields = vars(value)
        self._value = {key: Descriptor.describe(fields[key]) for key in filter(check_private, fields.keys())}

    def read(self):
        pass


def check_private(field: str) -> bool:
    if re.match('_', field):
        return False
    else:
        return True
