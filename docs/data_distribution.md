This file describes the different data distribution avalaible and how the data is splitted based on these data distributions.

## Data Distributions:
1. `iid` : If the data distribution used is `iid` the data will be equally partitioned among all clients i.e every client will recieve data from all the classes of the selected dataset.
2. `dirichlet_niid` : This is a relatively new approach to generate non-iid data based on the dirichlet distribution process. Some more details regarding this process can be obtained from here https://arxiv.org/abs/1909.06335 .