from util.util import save_metrics, get_param, initparams
import time
import os
import yaml
import traceback
import sys
import pandas as pd
from util import data_loader

frequency_map = {'yearly': 'Y', 'monthly': 'M', 'daily': 'D', 'quarterly': 'Q', 'weekly': 'W', 'hourly': 'H'}  # todo


def run(types, data_sizes, trail):
    # global type, path, f, config, metric, shape, task, run_kwargs
    print("1 initparams")
    data_base_path, report_base_path, max_trials, mode, vers = initparams()

    print("2 check for history run with the same name")
    trained_data_names = []
    white_list = []
    # sys.argv[1] report

    if len(sys.argv) > 1:
        result_file_path = report_base_path + os.sep + sys.argv[1] + '.csv'
    else:
        result_file_path = report_base_path + os.sep + 'benchmark_hyperts_' + types[0] + "_" + time.strftime(
            "%Y%m%d_%H%M%S",
            time.localtime()) + '.csv'
    if os.path.exists(result_file_path):
        trained_data_names = pd.read_csv(result_file_path)['dataset'].values

    # sys.argv[2] white_list
    if len(sys.argv) > 2:
        white_list = sys.argv[2].split(',')

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
                    if len(white_list) > 0 and dir not in white_list:
                        continue

                    train_file_path = path + os.sep + dir + os.sep + 'train.tsf'
                    if (os.path.exists(train_file_path) and os.path.getsize(train_file_path)):
                        print("train_file_path: ", train_file_path)
                        loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = data_loader.convert_tsf_to_dataframe(
                            train_file_path)

                        for i in range(loaded_data.shape[0]):
                            data_name = dir + '_' + loaded_data.loc[i]['series_name']

                            # if data_name in trained_data_names:
                            #     print('==skipped== already trained ', data_name)
                            #     continue
                            print('========== data_name:', data_name)
                            df_train = pd.DataFrame(loaded_data.loc[i]['series_value'])
                            df_train['datetime'] = pd.date_range(
                                start=loaded_data.loc[i]['start_timestamp']
                                , periods=df_train.shape[0], freq=frequency_map[frequency]
                            )

                            df_test = df_train[-forecast_horizon:]
                            df_train = df_train[:-forecast_horizon]

                            # df_train.to_csv(r'D:\文档\0 DAT\99 bak\data_tmp\\' + data_name + '_train.csv', index=False)
                            # df_test.to_csv(r'D:\文档\0 DAT\99 bak\data_tmp\\' + data_name + '_test.csv', index=False)
                            # continue

                            try:
                                hypertsmetric, hypertscost, run_kwargs, metric, task = trail(df_train, df_test,
                                                                                             max_trials)
                                save_metrics(data_name, metric, hypertsmetric, hypertscost, df_train.shape, data_size,
                                             task,
                                             forecast_horizon,
                                             run_kwargs, result_file_path)
                            except Exception:
                                traceback.print_exc()
                                print(" Error: " + train_file_path)
    time_end = time.time()
    print("end  ", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("total cost", time_end - time_start)
