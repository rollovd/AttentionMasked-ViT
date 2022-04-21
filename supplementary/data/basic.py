from torch.utils.data import Dataset

class BasicDataset(Dataset):
    augment = None
    params_string = None

    def __init__(self, cfg, augment, **kwargs):
        super(BasicDataset, self).__init__(**kwargs)

        self.augment = augment
        self.params_string = []

        for key, val in cfg.items():
            setattr(self, key, val)
            self.params_string.append(f'{key}: {val}')

        self.params_string = '\n\t\t\t\t'.join(self.params_string)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __str__(self):
        return f'{type(self).__name__}(length={self.__len__()},\n\t\t\t\t{self.params_string})'
