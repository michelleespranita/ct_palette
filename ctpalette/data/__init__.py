#################################################################
# extended and adapted from:
# https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models
#################################################################

from functools import partial
import numpy as np
import random

from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

import ctpalette.core.util as Util
from ctpalette.core.praser import init_obj


def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0 """
    '''create dataset and set random seed'''
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    phase_dataset, val_dataset = define_dataset(logger, opt)

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle 
    
    ''' create dataloader and validation dataloader '''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    print(f"length of {opt['phase']} dataloader", len(dataloader))
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! '''
    if opt['global_rank']==0 and val_dataset is not None:
        dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args',{}))
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args) 
        print("val dataloader exists")
        print("length of val dataloader", len(val_dataloader))
    else:
        val_dataloader = None
        print("val dataloader doesn't exist")
    return dataloader, val_dataloader


def define_dataset(logger, opt):
    ''' loading Dataset() class from given file's name '''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    phase_dataset = init_obj(dataset_opt, logger, default_file_name='ctpalette.data.dataset', init_type='Dataset')
    # if opt["phase"] == "test":
    #     subset_indices = random.sample(list(range(0, len(phase_dataset))), k=100)
    #     phase_dataset = Subset(phase_dataset, subset_indices)
    val_dataset_opt = opt['datasets']["val"]['which_dataset']
    val_dataset = init_obj(val_dataset_opt, logger, default_file_name='ctpalette.data.dataset', init_type='Dataset')
    
    data_len = len(phase_dataset)
    logger.info('Dataset for {} have {} samples.'.format(opt['phase'], data_len))
    if opt['phase'] == 'train':
        valid_len = len(val_dataset)
        logger.info('Dataset for {} have {} samples.'.format('val', valid_len))   
    return phase_dataset, val_dataset

def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets
