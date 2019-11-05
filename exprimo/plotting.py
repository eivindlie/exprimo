import matplotlib.pyplot as plt
import matplotlib


def plot_event_trace(events, show_transfer_lines=True, cmap='Accent'):
    op_done_events = [e for e in events if e.type == 'op_done']
    transfer_done_events = [e for e in events if e.type == 'transfer_done']
    batches = max(events, key=lambda e: e.batch).batch + 1
    devices = sorted(list({e.device for e in op_done_events}))
    run_time = max(events, key=lambda e: e.end_time).end_time

    fig, gnt = plt.subplots()

    gnt.set_ylim(0, 10 + 10 * len(devices))
    gnt.set_xlim(0, run_time)

    gnt.set_yticks([5 + 5 + 10 * i for i in range(len(devices))])
    gnt.set_yticklabels(devices)

    gnt.grid(True)

    cmap_func = matplotlib.cm.get_cmap(cmap, 2*batches)
    for event in op_done_events:
        color = cmap_func((2*event.batch + int(event.backward)) / 2*batches)
        device_index = devices.index(event.device)
        gnt.broken_barh([(event.start_time, event.end_time - event.start_time)], (5 + 2 + 10 * device_index, 6),
                        color=color)

    if show_transfer_lines:
        for event in transfer_done_events:
            from_device_index = devices.index(event.from_device)
            to_device_index = devices.index(event.to_device)
            gnt.plot([event.start_time, event.end_time], [10 + 10 * from_device_index, 10 + 10 * to_device_index],
                     color='red')

    plt.show()
