#
#! Hyperparameter wide, narrow, final setting runs


#! Input
#  Should take folder as input and plot results per dataset across all methods and seeds
# also once show accross datasets as some sort of generalization plot

# TODO this is currently only for final hpo setting I will have to come up with how to do this for HPO wide and narrow and maybe also how to visualize HP Importance in each setting
# TODO AUC Number calculation

# * Time vs Performance
# Area under Curve Plots
# Each with RAW times and once with times that are normalized into [0,1] range
# Plot once for train_acc and once for val_acc per dataset
# Plot once for train_acc and once for val_acc across datasets
#! One Stage
# ? Time
# Runtime
# ? Accuracy (train/val)
# train_acc, val_acc

#! Two Stage
# ? Time
# For first "epoch" (e.g. all epochs of stage1 and first epoch of stage2): SUM(Querried Training Times that incoporates computational factor per dataset) * 200 + first runtime entry
# For following "epochs" (e.g. other epochs of stage2): next runtime entries
# ? Accuracy (train/val)
# train_acc, val_acc corresponding to the runtime entries (neglect corresponding to train_time)

#! Random Sampling (random search with one epoch) / Random Search
# ? Time
# Querried Training Time that incoporates computational factor per dataset * 200
# ? Accuracy (train/val)
# train_acc, val_acc

# * Time vs Stability
# Area between between Seeds of a Method Plots
# Each with RAW times and once with times that are normalized into [0,1] range
# Plot once for train_acc and once for val_acc per dataset
# Plot once for train_acc and once for val_acc across datasets

#! Use the same way of getting Time and Accuracy as above
