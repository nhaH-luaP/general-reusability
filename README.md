# SDAL
Sustainable Deep Active Learning aims at minimizing labeling costs while maximizing the performance of not just the querying model but any model.

## Files and their explanation
initialize.py -> Initialize and save model weights and initial labeled pools in order to ensure no additional randomness accross multiple machines
pretrain.py -> Pretrain Models using SimCLR algorithm
active_learning.py -> Classical Deep Active Learning
evaluate_pool.py -> Classical Deep Learning Training on a pool acquired from a DAL experiment
utils.py -> Useful functions for other files.