import sys

from exprimo.optimize import optimize_with_config


config_path = 'configs/experiments/e4_ga-malvik-resnet50.json'

if __name__ == '__main__':
    if len(sys.argv):
        config_path = sys.argv[0]

    optimize_with_config(config_path, set_log_dir=True)