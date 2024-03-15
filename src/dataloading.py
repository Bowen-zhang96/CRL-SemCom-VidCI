from torch.utils.data import DataLoader
from src.dataio import NFS_Video

REMOTE_CUSTOM_PATH = '/media/data6/cindy/custom_data'
LOCAL_CUSTOM_PATH = '/home/cindy/PycharmProjects/custom_data'

block_sz = 256


def loadTrainingDataset(args, color=False, test=False):
    # global REMOTE_CUSTOM_PATH
    # if args.ares:
    #     REMOTE_CUSTOM_PATH = '/media/data4b/cindy/custom_data'
    #
    # if not args.local:
    #     args.data_root = f'{REMOTE_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    # else:
    #     args.data_root = f'{LOCAL_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    data_root = f'{args.data_root}/nfs_block_rgb_{block_sz}_8f'
    if args.test:  # use a smaller dataset when you're testing
        split = 'test'
    else:
        split = 'train'
    train_dataset = NFS_Video(log_root=data_root,
                                  block_size=args.block_size,
                                  gt_index=args.gt,
                                  color=color,
                                  split=split,
                                  test=test)
    if test:
        return DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=False)
    return DataLoader(train_dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=True,
                      pin_memory=False,prefetch_factor=16)


def loadValDataset(args, color=False):
    # global REMOTE_CUSTOM_PATH
    # if args.ares:
    #     REMOTE_CUSTOM_PATH = '/media/data4b/cindy/custom_data'
    # if not args.local:
    #     args.data_root = f'{REMOTE_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    # else:
    #     args.data_root = f'{LOCAL_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    data_root = f'{args.data_root}/nfs_block_rgb_{block_sz}_8f'

    if args.test:
        return None
    else:
        split = 'test'
    val_dataset = NFS_Video(log_root=data_root,
                                block_size=args.block_size,
                                gt_index=args.gt,
                                color=color,
                                split=split)

    return DataLoader(val_dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=False,
                      pin_memory=False,prefetch_factor=16), val_dataset
