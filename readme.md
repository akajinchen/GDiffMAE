# GDiffMAE
Implementation of our  paper "GDiffMAE: Guided Diffusion Enhanced Mask
Graph AutoEncoder for Recommendation".
<!-- PDF file is here: https://le-wu.com/files/Publications/CONFERENCES/SIGIR-23-yang.pdf -->
![](framework.pdf)

This paper explores the untapped potential of generative SSL for graph-based recommender systems. We highlight two critical challenges: firstly, designing effective diffusion mechanisms to enhance semantic information and collaborative signals while avoiding optimization biases; and secondly, developing adaptive structural masking mechanisms within graph diffusion to improve overall model performance. Motivated by these challenges, we propose a novel approach: the Guided Diffusion enhanced Mask graph AutoEncoder (GDiffMAE). GDiffMAE integrates an adaptive mask encoder for structural reconstruction and a guided diffusion model for semantic reconstruction, addressing the limitations of current methods.

Prerequisites
-------------
* numpy==1.24.4
* scipy==1.15.3
* setproctitle==1.3.3
* torch==2.0.0+cu118
* torch_geometric==2.3.1
* torch_sparse==0.6.17



Usage
------
cd GDiffMAE
* python Main.py --data ml-1m


