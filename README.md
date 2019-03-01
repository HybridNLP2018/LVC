# LVC (Language-Visual Correspondance)
## Notebooks and materials to recreate the Look, Read and Enrich -Learning from Scientific Figures and their Captions paper.

Automatically interpreting scientific figures is particularly hard compared to natural images. However, there is a valuable source of information in scientific literature that until now has remained untapped: the correspondence between a figure and its caption. Here we 
introduce a Language-Vision Correspondence learning task that results from investigating what can be learnt by looking at a large number of figures and reading their captions and provide the necessary code and data to reproduce the experiments related to such task.

LVC trains visual and language networks without additional supervision other than pairs of unconstrained figures and captions. We also support transferring lexical and semantic knowledge from existing knowledge graphs, which has proved to significantly improve the resulting features. 

This repository also provides code and data to leverage the LVC visual and language features in transfer learning tasks involving scientific text and figures, namely classification and multi-modal machine comprehension. Upon execution, our experiments show improvement or results on par with supervised baselines and ad-hoc approaches.

## How to run the notebooks:
1. Execute the download script with the options you want to choose: 
```
python download.py --weights --embeddings --cross-scigraph --cross-semantic-scholar --cat-figures --cat-captions --tqa
```
2. Use the different notebooks to execute the experiments.


## Requirements:
300 GB of free space disk to reproduce all the experiments:
- cross-scigraph -> 45 GB
- cross-semantic-scholar -> 225 GB
- cat-figures -> 10 GB
- cat-captions -> 10 GB
- tqa -> 10 GB
