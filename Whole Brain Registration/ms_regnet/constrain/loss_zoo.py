"""
define the constrain and loss function
"""


class LossZoo:
    constrain2loss = {}

    @staticmethod
    def get_loss_by_constrain(constrain: str):

        '''
        if constrain type is not assigned,
        if the constrain has only one loss, that loss is selected
        or mse will be selected
        '''
        keys = [i for i in LossZoo.constrain2loss[constrain].keys()]
        if len(keys) == 1 or "mse" not in keys:
            print(f"constrain: {constrain} uses the {keys[0]} loss")
            return LossZoo.constrain2loss[constrain][keys[0]]
        else:
            print(f"constrain: {constrain} uses the mse loss")
            return LossZoo.constrain2loss[constrain]["mse"]

    @staticmethod
    def get_loss_by_constrain_and_type(constrain: str, type: str):
        print(f"constrain: {constrain} uses the {type} loss")
        return LossZoo.constrain2loss[constrain][type]

    @staticmethod
    def register(*args):
        def inner_register(func):
            for arg in args:
                if isinstance(arg, tuple) or isinstance(arg, list):
                    print(f"add loss {arg} {func} to loss zoo")
                    if LossZoo.constrain2loss.get(arg[0]) is None:
                        LossZoo.constrain2loss[arg[0]] = {arg[1]: func}
                    else:
                        LossZoo.constrain2loss[arg[0]][arg[1]] = func
                else:
                    print(f"add loss {arg} {func} to loss zoo")
                    LossZoo.constrain2loss[arg] = func
            return func

        return inner_register
