import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

results_file = '../../experiment_results/2020-03-24_malvik_resnet_1500sim_100real.csv'
plot_title = f'ResNet on Malvik - 1500 simulated, 100 benchmarked generations'

data = pd.read_csv(results_file, index_col=0, skiprows=0, names=['generation', 'batch_time'])

data.plot(legend=None)
plt.title(plot_title)
plt.xlabel('Generation')
plt.ylabel('Batch training time (ms)')
plt.show()

