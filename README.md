# LVC (Language-Visual Correspondance)
## Notebooks and materials to recreate the Look, Read and Enrich -Learning from Scientific Figures and their Captions paper.

Automatically interpreting scientific figures is par-ticularly hard compared to natural images.   How-ever,  there  is  a  valuable  source  of  information  inscientific  literature  that  until  now  has  remaineduntapped:   the  correspondence  between  a  figureand its caption.   Here we investigate what can belearnt by looking at a large number of figures andreading  their  captions  and  introduce  a  Language-Vision  Correspondence  learning  task  that  makesuse  of  our  observations.Training  visual  andlanguage networks without additional supervisionother than pairs of unconstrained figures and cap-tions is shown to successfully solve this task.  Wealso  demonstrate  that  transferring  lexical  and  se-mantic knowledge from existing knowledge graphssignificantly  improves  the  resulting  features.   Fi-nally,  we  show  how  such  features  can  be  usedin other tasks involving scientific text and figures,like classification and multi-modal machine com-prehension,  outperforming  or  on  par  with  super-vised baselines and ad-hoc approaches.

## How to run the notebooks:
1. Execute the download script with the options you want to choose: 
```
python download.py --weights --embeddings --cross-scigraph --cross-semantic-scholar --cat-figures --cat-captions --tqa
```
2. Use the different notebooks to execute the experiments.
