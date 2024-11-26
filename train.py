import torch
import model
import yaml
import argparse
# from utils import argparser
from accelerate import utils
import warnings
warnings.filterwarnings('ignore')
def main(configs):
    CD_framework=model.Change_Detection_Framework(config=configs)
    CD_framework.training_CD()
def get_argparser():
    parser = argparse.ArgumentParser(
                    prog='Change_Detection_Framework',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--config',type=str,default= '/private/icme_cd/UCD/configs/BIT_LEVIRCD.yml',help="Directory of the config file")
    return parser
############# accelerate launch train.py

if __name__=="__main__":
    utils.set_seed(8888)

    args=get_argparser().parse_args()

    with open(args.config,'r') as f:
        configs=yaml.safe_load(f)
        print(configs)
    main(configs)
