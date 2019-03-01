import requests
from tqdm import tqdm
import math
import zipfile
from optparse import OptionParser
import lzma
import tarfile

def download_weights():
    url = "https://zenodo.org/record/2577875/files/weights.tar.xz"
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0 
    with open("weights.tar.xz", 'wb') as f:
      for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', desc = "weights.tar.xz", leave = True):
        wrote = wrote  + len(data)
        f.write(data)
    if total_size != 0 and wrote != total_size:
      print("ERROR, something went wrong")
    f.close
    print("Unzipping... this process might take several minutes to complete.")
    with lzma.open("weights.tar.xz") as f:
        with tarfile.open(fileobj=f) as tar:
            content = tar.extractall('.')
    print("Done")
def download_cross_sci():
    url = "https://zenodo.org/record/2577875/files/cross-corpus-scigraph.tar.xz"
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0 
    with open("cross-corpus-scigraph.tar.xz", 'wb') as f:
      for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', desc = "cross-corpus-scigraph.tar.xz", leave = True):
        wrote = wrote  + len(data)
        f.write(data)
    if total_size != 0 and wrote != total_size:
      print("ERROR, something went wrong")
    f.close
    print("Unzipping... this process might take several minutes to complete.")
    with lzma.open("cross-corpus-scigraph.tar.xz") as f:
        with tarfile.open(fileobj=f) as tar:
            content = tar.extractall('./databases/')
    print("Done")
def download_cross_semantic():
    url = "https://zenodo.org/record/2577875/files/cross-corpus-semantic-scholar.tar.xz"
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0 
    with open("cross-corpus-semantic-scholar.tar.xz", 'wb') as f:
      for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', desc = "cross-corpus-semantic-scholar.tar.xz", leave = True):
        wrote = wrote  + len(data)
        f.write(data)
    if total_size != 0 and wrote != total_size:
      print("ERROR, something went wrong")
    f.close
    print("Unzipping... this process might take several minutes to complete.")
    with lzma.open("cross-corpus-semantic-scholar.tar.xz") as f:
        with tarfile.open(fileobj=f) as tar:
            content = tar.extractall('./databases/')
    print("Done")
def download_cat_captions():
    url = "https://zenodo.org/record/2577875/files/cat-corpus-captions.tar.xz"
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0 
    with open("cat-corpus-captions.tar.xz", 'wb') as f:
      for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', desc = "cat-corpus-captions.tar.xz", leave = True):
        wrote = wrote  + len(data)
        f.write(data)
    if total_size != 0 and wrote != total_size:
      print("ERROR, something went wrong")
    f.close
    print("Unzipping... this process might take several minutes to complete.")
    with lzma.open("cat-corpus-captions.tar.xz") as f:
        with tarfile.open(fileobj=f) as tar:
            content = tar.extractall('./databases/')
    print("Done")
def download_cat_figures():
    url = "https://zenodo.org/record/2577875/files/cat-corpus-figures.tar.xz"
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0 
    with open("cat-corpus-figures.tar.xz", 'wb') as f:
      for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', desc = "cat-corpus-figures.tar.xz", leave = True):
        wrote = wrote  + len(data)
        f.write(data)
    if total_size != 0 and wrote != total_size:
      print("ERROR, something went wrong")
    f.close
    print("Unzipping... this process might take several minutes to complete.")
    with lzma.open("cat-corpus-figures.tar.xz") as f:
        with tarfile.open(fileobj=f) as tar:
            content = tar.extractall('./databases/')
    print("Done")
def download_tqa():
    url = "https://s3.amazonaws.com/ai2-vision-textbook-dataset/dataset_releases/tqa/tqa_train_val_test.zip"
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0 
    with open("tqa_train_val_test.zip", 'wb') as f:
      for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', desc = "tqa_train_val_test.zip", leave = True):
        wrote = wrote  + len(data)
        f.write(data)
    if total_size != 0 and wrote != total_size:
      print("ERROR, something went wrong")
    f.close
    print("Unzipping... this process might take several minutes to complete.")
    zip_ref = zipfile.ZipFile("tqa_train_val_test.zip", 'r')
    zip_ref.extractall(".")
    zip_ref.close()
    print("Done")

def main():
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option("--cross-scigraph",
                      action="store_true",
                      dest="crossScigraph",
                      default=False,
                      help="Cross-modal experiment with Scigraph corpus")
    parser.add_option("--cross-semantic",
                      action="store_true",
                      dest="crossSemantic",
                      default=False,
                      help="Cross-modal experiment with Semantic Scholar corpus")
    parser.add_option("--cat-captions",
                      action="store_true",
                      dest="catCaptions",
                      default=False,
                      help="Categorization experiment with captions")
    parser.add_option("--cat-figures",
                      action="store_true",
                      dest="catFigures",
                      default=False,
                      help="Categorization experiment with figures")
    parser.add_option("--tqa",
                      action="store_true",
                      dest="tqa",
                      default=False,
                      help="TQA experiment")
    (options, args) = parser.parse_args()

    download_weights()

    if(options.crossScigraph):
        download_cross_sci()
    if(options.crossSemantic):
        download_cross_semantic()
    if(options.catCaptions):
        download_cat_captions()
    if(options.catFigures):
        download_cat_figures()
    if(options.tqa):
        download_tqa()

if __name__ == "__main__":
    main()