import time
import os

__version__ = '0.1'

PLOT_STYLE = 'darkgrid'

LOG_CONFIG = {
    'clear_files': True,
    'streams': [
        print,
        'output.log'
    ],
    'log_dir': '~/logs/default'
}


def set_log_dir(path):
    expanded_path = os.path.expanduser(path)
    if not os.path.exists(expanded_path):
        os.makedirs(expanded_path)

    LOG_CONFIG['log_dir'] = path


def get_log_dir():
    return os.path.expanduser(LOG_CONFIG['log_dir'])


set_log_dir(LOG_CONFIG['log_dir'])


if LOG_CONFIG['clear_files']:
    for stream in LOG_CONFIG['streams']:
        if isinstance(stream, str):
            with open(stream, 'w') as f:
                f.write('')


def log(*strings, end='\n', sep=''):
    for stream in LOG_CONFIG['streams']:
        if isinstance(stream, str):
            with open(os.path.expanduser(os.path.join(LOG_CONFIG['log_dir'], stream)), 'a') as f:
                timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
                string = sep.join(str(s) for s in strings)
                f.write(f'{timestamp}  {string}{end}')
        else:
            stream(*strings, end=end, sep=sep)

from exprimo.simulator import Simulator
from exprimo.plotting import plot_event_trace
from exprimo.device import DeviceGraph
from exprimo.graph import ComputationGraph
from exprimo.optimize import optimize_with_config
