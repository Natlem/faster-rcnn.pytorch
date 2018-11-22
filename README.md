# A *Faster* Pytorch Implementation of Faster R-CNN with Pruning

## Introduction

This is a fork of the original jwyang/faster-rcnn.pytorch Faster R-CNN code. This project add additional pruning to the Faster R-CNN like architecture in order to make it smaller and faster.

Currently there's 3 pruning techniques implemented:
- L1 Pruning (https://openreview.net/pdf?id=rJqFGTslg)
- Entropy Pruning (https://arxiv.org/abs/1706.05791)
- Molchanov (https://arxiv.org/abs/1611.06440)

## How-To
Upcoming

## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    } 
    
    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
