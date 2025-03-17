from naslib.utils import get_dataset_api
from naslib.search_spaces.core import Metric

from naslib.search_spaces import NasBench301SearchSpace
from naslib.search_spaces.nasbench301.conversions import convert_naslib_to_genotype as convert_naslib_nb301_to_genotype

########## START TODO ############
# TODO: 
# 1. Sample a random NAS-Bench-301 model
graph = NasBench301SearchSpace()
graph.sample_random_architecture()
graph.parse()
# 2. Get the NASLib and genotype representations of the model
graph.get_hash()
convert_naslib_nb301_to_genotype(graph)
# 3. Query the predicted performance of the model (loading the NB301 benchmark API might take some time)
benchmark_api = get_dataset_api(search_space='nasbench301', dataset='cifar10')
train_acc_parent = graph.query(metric=Metric.TRAIN_ACCURACY, dataset='cifar10', dataset_api=benchmark_api)
val_acc_parent = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=benchmark_api)
print('Performance of parent model')
print(f'Validation accuracy: {val_acc_parent:.2f}%')

# 4. Mutate the model
child_graph = NasBench301SearchSpace()

child_graph.mutate(parent=graph)
# 5. Get the NASLib and genotype representations of the model
child_graph.get_hash()
convert_naslib_nb301_to_genotype(child_graph)
# 6. Query the predicted performance of the child
val_acc_child = child_graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=benchmark_api)
print('Performance of child model')
print(f'Validation accuracy: {val_acc_child:.2f}%')