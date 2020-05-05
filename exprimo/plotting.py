import os
from collections import defaultdict
from glob import glob

import imageio as imageio
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.colors import Normalize
import seaborn as sns
from exprimo import PLOT_STYLE
sns.set(style=PLOT_STYLE)


def plot_event_trace(events, simulator, show_transfer_lines=True, show_memory_usage=True, cmap='Paired',
                     plot_op_time_distribution=False, save_path=None):
    op_done_events = [e for e in events if e.type == 'op_done']
    transfer_done_events = [e for e in events if e.type == 'transfer_done']
    batches = max(events, key=lambda e: e.batch).batch + 1
    devices = sorted(list({e.device for e in op_done_events}))
    run_time = max(events, key=lambda e: e.end_time).end_time

    fig, gnt = plt.subplots()

    gnt.set_ylabel('Device')
    gnt.set_xlabel('Time (ms)')

    gnt.set_ylim(0, 10 + 10 * len(devices))
    gnt.set_xlim(0, run_time)

    gnt.set_yticks([5 + 5 + 10 * i for i in range(len(devices))])
    gnt.set_yticklabels(devices)

    gnt.patch.set_alpha(0)

    gnt.grid(True)

    if show_memory_usage:
        _, memory_x, memory_y = simulator.calculate_memory_usage(events, return_memory_history=True)

        plot_height = 1 / (len(devices) + 1)
        for d, device in enumerate(devices):
            # max_memory = simulator.device_graph.devices[device].device.memory * 10**9
            ax = gnt.inset_axes((0, plot_height * (d + 1), 1, plot_height / 2), sharex=gnt, zorder=1)
            # ax.set_ylim(0, max_memory)
            ax.axis('off')
            ax.fill_between(memory_x, memory_y[device], facecolor='grey', alpha=0.2)
            ax.set(ylim=(0, None))

    cmap_func = matplotlib.cm.get_cmap(cmap, 2*batches)
    cmap_norm = Normalize(vmin=0, vmax=2*batches)
    for event in op_done_events:
        color = cmap_func(cmap_norm(2*event.batch + int(event.backward)))
        device_index = devices.index(event.device)
        gnt.broken_barh([(event.start_time, event.end_time - event.start_time)], (5 + 2 + 10 * device_index, 6),
                        color=color, zorder=10, alpha=0.9)

    if show_transfer_lines:
        for event in transfer_done_events:
            from_device_index = devices.index(event.from_device)
            to_device_index = devices.index(event.to_device)
            gnt.plot([event.start_time, event.end_time], [10 + 10 * from_device_index, 10 + 10 * to_device_index],
                     color='red', alpha=0.4)

    if save_path:
        plt.savefig(os.path.expanduser(save_path), bb_inches='tight')

    plt.show()
    fig.close(fig)

    if plot_op_time_distribution:
        op_times = defaultdict(lambda: 0)

        for event in op_done_events:
            op_times[event.op_name] += event.end_time - event.start_time

        plt.figure(figsize=(15, 10))
        plt.xticks(rotation='vertical')
        plt.bar(op_times.keys(), op_times.values())
        plt.show()


def plot_map_elites_archive(archive_scores, n_devices=None, max_jumps=None, axes=(1, 2), save_path=None,
                            return_fig=False, vmin=None, vmax=None, title=None):
    assert min(axes) >= 0 and max(axes) < len(archive_scores.shape), 'Axes out of range!'

    dimension_sizes = archive_scores.shape
    if n_devices == None:
        n_devices = dimension_sizes[0]

    if max_jumps == None:
        max_jumps = dimension_sizes[2]

    AXIS_NAMES = ['Most common device', 'No. of used devices', 'No. of jumps']
    AXIS_TICKS = [
        [int(i * (n_devices / dimension_sizes[0])) for i in range(dimension_sizes[0])],
        [int(i * (n_devices / dimension_sizes[1])) + 1 for i in range(dimension_sizes[1])],
        [int(i * (max_jumps / dimension_sizes[2])) for i in range(dimension_sizes[2])]
    ]

    if len(axes) < len(archive_scores.shape):
        avg_axes = [i for i in range(len(archive_scores.shape)) if i not in axes]
        avg_batch_times = 1 / np.nanmean(archive_scores, axis=tuple(avg_axes))
    else:
        avg_batch_times = 1 / archive_scores

    mask = np.isnan(avg_batch_times)

    min_time = np.nanmin(avg_batch_times)
    max_time = np.nanmax(avg_batch_times)

    n_plots = int(len(axes) == 2 or archive_scores.shape[0])

    plotted_axes = tuple(reversed(sorted(ax for ax in axes if len(axes) == 2 or ax != 0)))

    figsize = (10 * min(3, n_plots), 8 * ((n_plots - 1) // 3 + 1))
    fig, axs = plt.subplots((n_plots - 1) // 3 + 1, min(3, n_plots), figsize=figsize)
    axs = np.reshape(axs, (-1,))
    for i, ax in enumerate(axs):
        if i >= avg_batch_times.shape[0]:
            ax.axis('off')
            break

        cmap = sns.cm.rocket_r
        data = avg_batch_times[i, :, :] if len(axes) > 2 else avg_batch_times
        mask1 = mask[i, :, :] if len(axes) > 2 else mask
        if np.all(mask1):
            ax.axis('off')
            continue
        plot = sns.heatmap(data, ax=ax, mask=mask1, square=True, cmap=cmap,
                           xticklabels=AXIS_TICKS[plotted_axes[0]], yticklabels=AXIS_TICKS[plotted_axes[1]],
                           vmin=vmin if vmin else min_time, vmax=vmax if vmax else max_time)
        plot.invert_yaxis()

        ax.set_xlabel(AXIS_NAMES[plotted_axes[0]])
        ax.set_ylabel(AXIS_NAMES[plotted_axes[1]])
        if len(axes) > 2:
            ax.set_title(f'{AXIS_NAMES[0]} = {AXIS_TICKS[0][i]}', pad=80)

    if title:
        fig.suptitle(title)

    if save_path:
        plt.savefig(os.path.expanduser(save_path), bb_inches='tight')

    if return_fig:
        return fig

    plt.show()


def plot_archive_animation(paths, save_path, dimension_sizes, n_devices=None, max_jumps=None, axes=(1, 2), fps=1):
    archives = []

    max_time = -np.inf
    min_time = np.inf

    step_names = []

    paths = sorted(paths)

    for path in paths:
        file_name = path.split('/')[-1]
        step = file_name.replace('step_', '').replace('.csv', '')
        step_names.append(step)
        archive = np.empty(dimension_sizes)
        archive[:] = np.NaN
        with open(path) as f:
            for l, line in enumerate(f):
                if l == 0: # Skip header row
                    continue
                niche, time, _ = line.split(';')
                niche = eval(niche)
                time = float(time)

                max_time = max(max_time, time)
                min_time = min(min_time, time)

                archive[niche[0], niche[1], niche[2]] = 1 / time

        archives.append(archive)

    images = []

    for i, archive in enumerate(archives):
        fig = plot_map_elites_archive(archive, n_devices, max_jumps, axes, vmin=min_time, vmax=max_time,
                                      return_fig=True, title=f'Step {step_names[i]}')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close()
        images.append(image)

    if isinstance(save_path, list) or isinstance(save_path, tuple):
        for path in save_path:
            imageio.mimsave(path, images, fps=fps)
    else:
        imageio.mimsave(save_path, images, fps=fps)
