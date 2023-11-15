import argparse
import pathlib
import os

from knnbox.models.vae_mt import VAEForSimilarSimples, SampleGenerator, StopwatchMeter
from knnbox.datastore import Datastore

def get_absolute_root_dir():
    p = os.path.abspath(".")
    
    if os.path.exists(os.path.join(p, "train_phase1.py")):
        return os.path.split(os.path.split(p)[0])[0]
    elif os.path.isdir(os.path.join(p, "vae-mt")):
        return os.path.split(p)[0]
    elif os.path.exists(os.path.join(p, "knnbox-scripts/vae-mt/train_phase1.py")):
        return p
    else:
        raise RuntimeError("Can't determine where are u")
    
def get_args():
    ps = argparse.ArgumentParser()
    

if __name__ == "__main__":
    root_dir = get_absolute_root_dir()
    
    