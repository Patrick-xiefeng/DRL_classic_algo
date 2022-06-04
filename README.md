# Classic algorithms in Deep Reinforcement Learning
This repo aims to implement the classic deep reinforcement learning algorithms in Pytorch version.
# Installation
1. clone this repo. and we will call the directory that you cloned as ${DRL_classic_algo}
2. Install dependencies. We use python 3.9 and pytorch 1.11.0 stable version, and our CUDA version is 10.2.
```angular2html
conda create -n DRL 
conda activate DRL
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install gym[all]
conda install matplotlib
```
3. The results of each algorithm can be found in 'logs' directory and can be seen in tensorboard.

# Acknowledgement
A large part of the code is borrowed from [XinJingHao/RL-algorithms-by-Pytorch]('https://github.com/XinJingHao/RL-Algorithms-by-Pytorch') and
[higgsfield/RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2).

Thanks for their wonderful works.