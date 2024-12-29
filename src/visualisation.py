import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_time_series(lineages_time_series, n_lineages, n_steps):
  
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for lineage in lineages_time_series[:n_lineages]:
        axes[0].plot(lineage["t"][:n_steps], lineage["x"][:n_steps])
        axes[1].plot(lineage["t"][:n_steps], lineage["s"][:n_steps])

    axes[0].set_title('molecule number time series ')
    axes[1].set_title('cell size time series')
    axes[1].set_yscale('log')

    plt.show()



