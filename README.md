# Temporal Graph Q-Network
### Reinforcement learning recommender system based on temporal graph network 
**TGQN** is a recomendation system model based 
on graph neural network ([TGN](https://arxiv.org/abs/2006.10637)) and trained in RL-fashion by Q-learning.  

Design and research of the model was my masters project defeated at 
[MIPT](https://mipt.ru/english/). The main idea was to replace GCN in existing [GCQN](https://dl.acm.org/doi/abs/10.1145/3397271.3401237) model to TGN and 
set up experiment with comparisons TGQN vs GCQN and some classic recsys 
models(SVD and RNN4Rec) which also trained with Q-learning on three 
benchmark datasets.

Masters thesis can be found [here](https://github.com/nmineev/mipt-masters-project/blob/main/thesis/masters_thesis_final.pdf).  
Slides for defence can be found [here](https://github.com/nmineev/mipt-masters-project/blob/main/thesis/msproject_slides.pdf).  
Benchmark datasets can be found [here](https://github.com/nmineev/mipt-masters-project/blob/main/code/data/README.md).  
Code of the experiments can be found [here](https://github.com/nmineev/mipt-masters-project/tree/main/code).  
