import os


def load_log(path):
    log = {}
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            item_list = line.strip().split(' ')
            fold, epoch = tuple(map(int, item_list[0].split('-')))
            l_value = float(item_list[1].split('=')[1])
            d_value = float(item_list[2].split('=')[1])
            t_value = float(item_list[3].split('=')[1])
            dev_mse = float(item_list[4].split('=')[1])
            test_mse = float(item_list[5].split('=')[1])
            if fold not in log:
                log[fold] = {'epoch': [], 'loss': [], 'dev_acc': [], 'test_acc': [], 'dev_mse': [], 'test_mse': []}
            log[fold]['epoch'].append(epoch)
            log[fold]['loss'].append(l_value)
            log[fold]['dev_acc'].append(d_value)
            log[fold]['test_acc'].append(t_value)
            log[fold]['dev_mse'].append(dev_mse)
            log[fold]['test_mse'].append(test_mse)
    return log


def visualize(log_dir):
    experiments = {}
    for home, dirs, files in os.walk(log_dir):
        for f in files:
            exp_name = f.split('.', maxsplit=1)[0]
            if exp_name not in experiments:
                experiments[exp_name] = load_log(home + f)
            else:
                print('Duplicated experiment log')
    return experiments


def get_average(result):
    fout = open('./log/summary.log', 'w', encoding='utf-8')
    for k in result:
        test_avg = 0.0
        dev_avg = 0.0
        test_mse = 0.0
        dev_mse = 0.0
        for epoch in result[k].values():
            test_avg += max(epoch['test_acc'])
            dev_avg += max(epoch['dev_acc'])
            test_mse += min(epoch['test_mse'])
            dev_mse += min(epoch['dev_mse'])
        test_avg /= len(result[k])
        print('{}:\ttest_acc={:.4f}'.format(k, test_avg), file=fout)
        dev_avg /= len(result[k])
        print('{}:\tdev_acc={:.4f}'.format(k, dev_avg), file=fout)
        test_mse /= len(result[k])
        print('{}:\ttest_mse={:.6f}'.format(k, test_mse), file=fout)
        dev_mse /= len(result[k])
        print('{}:\tdev_mse={:.6f}\n'.format(k, dev_mse), file=fout)

    fout.close()


if __name__ == '__main__':
    result = visualize('./log/')
    get_average(result)
