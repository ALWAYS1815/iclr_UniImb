from torch.utils.data import Dataset as BaseDataset
from torch_geometric.data.collate import collate
import torch
from generator import *

class Dataset(BaseDataset):
    def __init__(self, dataset, all_dataset, args):
        self.args = args
        self.dataset = dataset
        self.all_dataset = all_dataset
        self.dataname = args.dataset
        self.pos_enc = args.pos_enc
    def _get_feed_dict(self, index):
        feed_dict = self.dataset[index]
        return feed_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):
        batch_id = torch.tensor([feed_dict.id for feed_dict in feed_dicts])
        train_idx = torch.arange(batch_id.shape[0])

        data, slices, _ = collate(
            feed_dicts[0].__class__,
            data_list=feed_dicts,
            increment=True,
            add_batch=True,
        )
        
        if len(feed_dicts) < self.args.batch_size:
            padding_size = self.args.batch_size - len(feed_dicts)
            last_graph = feed_dicts[-1]  
            for _ in range(padding_size):
                feed_dicts.append(last_graph)

            data, slices, _ = collate(
            feed_dicts[0].__class__,
            data_list=feed_dicts,
            increment=True,
            add_batch=True)
        batch = {'data': data, 'train_idx': train_idx}
        return batch
    
    
