from util.util import save_metrics, get_param, initparams
import time
import os
import yaml
import traceback
import sys
import pandas as pd


def run(types, data_sizes, trail):
    # global type, path, f, config, metric, shape, task, run_kwargs
    print("1 initparams")
    data_base_path, report_base_path, max_trials, mode, vers = initparams()

    print("2 check for history run with the same name")
    trained_data_names = []
    if len(sys.argv) > 1:
        result_file_path = report_base_path + os.sep + sys.argv[1] + '.csv'
    else:
        result_file_path = report_base_path + os.sep + 'benchmark_hyperts_' + types[0] + "_" + time.strftime(
            "%Y%m%d_%H%M%S",
            time.localtime()) + '.csv'
    if os.path.exists(result_file_path):
        trained_data_names = pd.read_csv(result_file_path)['dataset'].values

    print("3 start to run ------------ ")
    time_start = time.time()
    print("start", time.strftime("%Y-%m-%d %H:%M:%S"))
    for type in types:
        print("3 type:" + type)
        for data_size in data_sizes:
            print("4 data_size:" + data_size)
            path = data_base_path + os.sep + type + os.sep + data_size
            if os.path.exists(path):
                list = os.listdir(path)
                for dir in list:
                    if dir == '__init__.py' or dir == 'template':
                        continue

                    train_file_path = path + os.sep + dir + os.sep + 'train.csv'
                    if mode == 'dev':
                        train_file_path = path + os.sep + dir + os.sep + 'train_dev.csv'
                    test_file_path = path + os.sep + dir + os.sep + 'test.csv'
                    metadata_path = path + os.sep + dir + os.sep + 'metadata.yaml'

                    if (os.path.exists(train_file_path) and os.path.getsize(train_file_path))\
                            or (os.path.exists(train_file_path[0:-4] + '.pkl') and os.path.getsize(train_file_path[0:-4] + '.pkl')) > 0:
                        print("train_file_path: ", train_file_path)
                        print("test_file_path: ", test_file_path)
                        print("metadata_path: ", metadata_path)
                        f = open(metadata_path, 'r', encoding='utf-8')
                        config = yaml.load(f.read(), Loader=yaml.FullLoader)
                        forecast_len = get_param(config, 'forecast_len')
                        # type = config['type']
                        dtformat = get_param(config, 'dtformat')
                        date_col_name = get_param(config, 'date_col_name')
                        data_name = config['name']
                        metric = config['metric']
                        shape = config['shape']
                        task = config['task']
                        forecast_len = get_param(config, 'forecast_len')
                        series_col_name = config['series_col_name'].split(",") if 'series_col_name' in config else None
                        covariables = config['covariables_col_name'].split(
                            ",") if 'covariables_col_name' in config else None
                        f.close()

                        if data_name in trained_data_names:
                            print('==skipped== already trained ', data_name)
                            continue

                        try:
                            hypertsmetric, hypertscost, run_kwargs = trail(train_file_path, test_file_path,
                                                                           date_col_name,
                                                                           series_col_name, forecast_len, dtformat,
                                                                           type,
                                                                           metric,
                                                                           covariables, max_trials)
                            save_metrics(data_name, metric, hypertsmetric, hypertscost, shape, data_size, task,
                                         forecast_len,
                                         run_kwargs, result_file_path)
                        except Exception:
                            traceback.print_exc()
                            print(" Error: " + train_file_path)
                        time.sleep(1)
    time_end = time.time()
    print("end  ", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("total cost", time_end - time_start)
