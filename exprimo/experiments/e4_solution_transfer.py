from exprimo.optimize import optimize_with_config


config_path = 'configs/experiments/e4-4_ga-malvik-resnet50_long.json'

if __name__ == '__main__':
    optimize_with_config(config_path, set_log_dir=True)