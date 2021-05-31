from model.MiningModelElement import MiningModelElement


class DenseLayer(MiningModelElement):
    def __init__(self, weights, activation_function, number_of_inputs, number_of_outputs, bias, use_bias):
        self.weights = weights
        self.activation_function = activation_function
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.bias = bias
        self.use_bias = use_bias

    @staticmethod
    def from_proto(proto_layer):
        proto_layer = proto_layer.object.fields['properties'].list.descriptors
        weights = proto_layer[0]
        number_of_inputs = proto_layer[1].int_value
        number_of_outputs = proto_layer[2].int_value
        bias = proto_layer[3]
        use_bias = proto_layer[4].bool_value
        activation_function = proto_layer[5].enumeration.enum_value_name
        return DenseLayer(
            weights=weights,
            number_of_inputs=number_of_inputs,
            number_of_outputs=number_of_outputs,
            bias=bias,
            use_bias=use_bias,
            activation_function=activation_function
        )
