class Optimizer(object):
    def __init__(self, options):
        self.options = options
    
    def optimizer(self, fitness_function):
        pass
    
    def __repr__(self):
        tip = "NOTE that the optimizer'name to be printed is not set. " +\
            "Set the field 'optimizer_name' for the dict object 'options'."
        optimizer_name = self.options.get("optimizer_name", tip)
        return optimizer_name
