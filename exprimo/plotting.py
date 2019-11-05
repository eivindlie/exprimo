import matplotlib.pyplot as plt


def plot_event_trace(events):
    op_done_events = [e for e in events if e.type == 'op_done']
    devices = sorted(list({e.device for e in op_done_events}))
    run_time = max(events, key=lambda e: e.end_time).end_time

    fig, gnt = plt.subplots()

    gnt.set_ylim(0, 10 + 10 * len(devices))
    gnt.set_xlim(0, run_time)

    gnt.set_yticks([5 + 5 + 10 * i for i in range(len(devices))])
    gnt.set_yticklabels(devices)

    gnt.grid(True)

    for event in op_done_events:
        device_index = devices.index(event.device)
        gnt.broken_barh([(event.start_time, event.end_time - event.start_time)], (5 + 2 + 10 * device_index, 6))

    plt.show()