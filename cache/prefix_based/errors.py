class MissingKwargError(TypeError):
    def __init__(self, arg):
        super().__init__(f'Missing keyword argument {arg}')
