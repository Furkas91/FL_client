from model.MiningModelElement import MiningModelElement


class Connection(MiningModelElement):
    def __init__(self, first, second, weight):
        self.first = first
        self.second = second
        self.weight = weight

    def get_output(self):
        pass

    def get_weighted_output(self):
        pass

    def merge(self):
        pass