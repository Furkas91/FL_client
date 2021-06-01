from torch import nn

loss_functions = {
    'NEGATIVE_LOG_LIKELIHOOD': nn.NLLLoss,
    'MEAN_SQUARE_ERROR': nn.MSELoss
}


class MiningSettings:
    def __init__(self, algorithm, loss_function, epochs, learning_rate, momentum, batch_size):
        self.algorithm = algorithm
        self.loss_function = loss_functions[loss_function]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

    @staticmethod
    def from_proto(proto_settings):
        proto_settings = proto_settings.object.fields['algorithmSettings'].object.fields
        algorithm = proto_settings['algorithmName'].string_value
        settings = proto_settings['settings'].map.entries
        loss_function = settings[0].value.enumeration.enum_value_name
        epochs = settings[1].value.int_value
        batch_size = settings[2].value.int_value
        momentum = settings[3].value.double_value
        learning_rate = settings[4].value.double_value


        return MiningSettings(algorithm=algorithm,
                              loss_function=loss_function,
                              epochs=epochs,
                              learning_rate=learning_rate,
                              momentum=momentum,
                              batch_size=batch_size)