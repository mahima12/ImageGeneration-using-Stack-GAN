from __future__ import division
from __future__ import print_function

import dateutil
import dateutil.tz
import datetime
import argparse
import pprint

from stageII.misc.datasets import TextDataset
#from stageI.model import CondGAN
from model import CondGAN
#from stageI.trainer import CondGANTrainer
from trainer import CondGANTrainer
from stageII.misc.utils import mkdir_p
from stageII.misc.config import cfg, cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    print('Using config:')
    #data pretty printer
    pprint.pprint(cfg)
    print("11111111111")
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    datadir = 'Data/%s' % cfg.DATASET_NAME
    print("datadir "+ datadir)
    dataset = TextDataset(datadir, cfg.EMBEDDING_TYPE, 1)
    print("dataset "+ str(dataset))
    filename_test = '%s/test' % (datadir)
    print("filename_test "+ filename_test)
    dataset.test = dataset.get_data(filename_test)
    if cfg.TRAIN.FLAG:
        print("2222222222")
        filename_train = '%s/train' % (datadir)
        print("filename_train "+ filename_train)
        dataset.train = dataset.get_data(filename_train)
       
        ckt_logs_dir = "ckt_logs/%s/%s_%s" % \
            (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(ckt_logs_dir)
    else:
        print("3333333333")
        s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        ckt_logs_dir = s_tmp[:s_tmp.find('.ckpt')]
    print("Inside CONDGAN")
    model = CondGAN(
        image_shape=dataset.image_shape
    )
    print("outside CONDGAN")
    print("inside CONDGANTrainer")
    algo = CondGANTrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=ckt_logs_dir
    )
    print("outside CONDGANTrainer")
    if cfg.TRAIN.FLAG:
        print("inside algo")
        algo.train()
    else:
        ''' For every input text embedding/sentence in the
        training and test datasets, generate cfg.TRAIN.NUM_COPY
        images with randomness from noise z and conditioning augmentation.'''
        algo.evaluate()
