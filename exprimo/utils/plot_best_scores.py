import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')

results_file = '../../experiment_results/2020-04-07_malvik_resnet_200sim_100real_limited-memory.csv'
plot_title = f'ResNet on Malvik - 200 simulated, 100 benchmarked generations\nLimited memory'

benchmark_limit = 200

data = pd.read_csv(results_file, index_col=0, skiprows=0, names=['generation', 'batch_time'])

data.plot(legend=None)

if benchmark_limit:
    plt.axvline(x=benchmark_limit, c='grey', ls='--')

plt.title(plot_title)
plt.xlabel('Generation')
plt.ylabel('Batch training time (ms)')
plt.show()

