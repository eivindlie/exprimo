import os
from datetime import datetime

import GPUtil
import psutil

from exprimo import get_log_dir


def clear_processor_log():
    with open(os.path.join(get_log_dir(), 'processor_util.csv'), 'w') as f:
        headers = ['timestamp', 'step']
        for gpu in range(len(GPUtil.getGPUs())):
            headers.append(f'gpu:{gpu}')

        for cpu in range(psutil.cpu_count()):
            headers.append(f'cpu:{cpu}')

        f.write(f'{",".join(headers)}\n')


def update_processor_log(step='null'):
    log_file = os.path.join(get_log_dir(), 'processor_util.csv')
    if not os.path.exists(log_file):
        clear_processor_log()

    with open(log_file, 'a') as f:
        log_line = [
            datetime.now().isoformat(),
            step
        ]

        for gpu in GPUtil.getGPUs():
            log_line.append(f'{gpu.load*100:.2f}')

        for cpu_util in psutil.cpu_percent(percpu=True):
            log_line.append(f'{cpu_util:.2f}')

        f.write(f'{",".join(map(str, log_line))}\n')


if __name__ == '__main__':
    update_processor_log()
