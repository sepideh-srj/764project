# Mix and Match 
This is a Pytorch implementation of our mix and match project. 
Parts from different chairs are selected randomly, merged and given as input to the SCORES network.
SCORES is a recursive neural network (RvNN).
Baseline of our project is as follows:
![alt text](https://github.com/sepideh-srj/764project/blob/main/cg.jpg)

### pre-process
To make the different parts of legs, backs and arms more consistant, they are merged into a single box.
### post-process
Transformed parts may have small misalignments at connection points, to make the matching between different parts better we make sure that boxes from different parts connect to each other.

## Usage
This implementation should be run with Python 3.x and Pytorch 0.4.0.


## Description

## Reference
```
@article {zhu_siga18,
    title = {SCORES: Shape Composition with Recursive Substructure Priors},
    author = {Chenyang Zhu and Kai Xu and Siddhartha Chaudhuri and Renjiao Yi and Hao Zhang},
    journal = {ACM Transactions on Graphics (SIGGRAPH Asia 2018)},
    volume = {37},
    number = {6},
    pages = {to appear},
    year = {2018}
}
```
