from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
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
        plt.savefig(save_path)

    plt.show()

    if plot_op_time_distribution:
        op_times = defaultdict(lambda: 0)

        for event in op_done_events:
            op_times[event.op_name] += event.end_time - event.start_time

        plt.figure(figsize=(15, 10))
        plt.xticks(rotation='vertical')
        plt.bar(op_times.keys(), op_times.values())
        plt.show()
