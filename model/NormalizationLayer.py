from model.MiningModelElement import MiningModelElement


class NormalizationLayer1(MiningModelElement):
    @staticmethod
    def from_proto(proto_layer):
        return NormalizationLayer1()
