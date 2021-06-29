"""
Файл, описывающий абстрактный класс адаптера
"""
from abc import ABC, abstractmethod


class Adapter(ABC):

    @staticmethod
    @abstractmethod
    def from_proto(proto):
        pass

    @abstractmethod
    def to_proto(self):
        pass
