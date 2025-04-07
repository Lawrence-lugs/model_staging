import torch

class DataloaderTuple:
    '''
    Transforms a dataloader from one that outputs a dict -- huggingface-style 
    to one that outputs a tuple of (img, label) -- classic pytorch-style
    '''
    def __init__(self, dataset, **kwargs):
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            **kwargs
        )
        self.batch_size = self.dataloader.batch_size

    def __iter__(self):
        return self._generate_data()

    def _generate_data(self):
        for batch in self.dataloader:
            yield batch["img"], batch["label"]