import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
        
    elif opt.dataset_mode == 'aligned_rand':
        from data.aligned_dataset_rand import AlignedDataset_Rand
        dataset = AlignedDataset_Rand()
        
    elif opt.dataset_mode == 'aligned_test':
        from data.aligned_dataset_test import AlignedDataset_Test
        dataset = AlignedDataset_Test()

    elif opt.dataset_mode == 'unaligned_seg':
        from data.unaligned_dataset_seg import UnalignedDataset_Seg
        dataset = UnalignedDataset_Seg()

    elif opt.dataset_mode == 'aligned_seg':
        from data.aligned_dataset_seg import AlignedDataset_Seg
        dataset = AlignedDataset_Seg()
    elif opt.dataset_mode == 'aligned_seg_rand':
        from data.aligned_dataset_seg_rand import AlignedDataset_Seg_Rand
        dataset = AlignedDataset_Seg_Rand()
        
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
        
    elif opt.dataset_mode == 'fivek':
        from data.fivek_dataset import FiveKDataset
        dataset = FiveKDataset()

    elif opt.dataset_mode == 'fivek2':
        from data.fivek_dataset2 import FiveKDataset2
        dataset = FiveKDataset2()

    elif opt.dataset_mode == 'fivek3':
        from data.fivek_dataset3 import FiveKDataset3
        dataset = FiveKDataset3()
    elif opt.dataset_mode == 'fivek4':
        from data.fivek_dataset4 import FiveKDataset4
        dataset = FiveKDataset4()
    elif opt.dataset_mode == 'fivek4_syn':
        from data.fivek_dataset4_syn import FiveKDataset4_syn
        dataset = FiveKDataset4_syn()
    elif opt.dataset_mode == 'fivek_single':
        from data.fivek_single import FiveKDataset_single
        dataset = FiveKDataset_single()
    
    elif opt.dataset_mode == 'ava':
        from data.ava_dataset import AVADataset
        dataset = AVADataset()

    elif opt.dataset_mode == 'aadb':
        from data.aadb_dataset import AADBDataset
        dataset = AADBDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
