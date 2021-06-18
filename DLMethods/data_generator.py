import json
import os
import random
import pandas as pds
from imblearn.over_sampling import BorderlineSMOTE

import numpy as np

import src.utils as utils


class DataGenerator:
    def __init__(self, data_dir, batch_size, conv=False):
        self.n_fold = 10
        self.batch_size = batch_size
        self.conv = conv
        self.channels = 1
        self.cache_name = 'train_complete.cache'
        self.submit_name = 'test_complete.cache'
        self.data_dir = data_dir

        self.data = self.load_data()
        self.test_pos = 0
        self.round_end = False
        self.test_data = self.data[self.test_pos]
        self.train_data = self.get_train_data()
        self.dev_pos = self.get_dev_pos()
        self.dev_data = self.data[self.dev_pos]
        self.emb_size = len(self.data[0][0][1])
        print('Train set size: {}'.format(len(self.train_data)))
        print('Dev set size: {}'.format(len(self.dev_data)))
        print('Test set size: {}'.format(len(self.test_data)))
        print('Embedding size: {}\n'.format(self.emb_size))

        self.null_column_names = None
        self.submit_data = None

    def train_num(self):
        return len(self.train_data)

    def test_num(self):
        return len(self.test_data)

    def dev_num(self):
        return len(self.dev_data)

    def get_dev_pos(self):
        dev_pos = random.randrange(0, self.n_fold)
        while dev_pos == self.test_pos:
            dev_pos = random.randrange(0, self.n_fold)
        return dev_pos

    def new_round(self):
        self.round_end = False

    def get_train_data(self):
        train_set = []
        for i in range(len(self.data)):
            if not i == self.test_pos:
                train_set.extend(self.data[i])
        return train_set

    def switch_train_test(self):
        self.test_pos += 1
        if self.test_pos == self.n_fold:
            self.round_end = True
        self.test_pos = self.test_pos % self.n_fold
        self.test_data = self.data[self.test_pos]
        self.train_data = self.get_train_data()
        self.dev_pos = self.get_dev_pos()
        self.dev_data = self.data[self.dev_pos]

    def smote_balance(self, data):
        smote = BorderlineSMOTE(random_state=2021)
        labels, examples = zip(*data)
        new_examples, new_labels = smote.fit_resample(examples, labels)
        return list(zip(new_labels.tolist(), new_examples.tolist()))

    def data_balance(self, data):
        balanced = []
        labels, examples = zip(*data)
        label_count = {}
        for i, label in enumerate(labels):
            if label not in label_count:
                label_count[label] = [examples[i]]
            else:
                label_count[label].append(examples[i])
        max_count = max([len(e) for e in label_count.values()])
        for label in label_count:
            filled_data = self.generate_disturbed_data(label_count[label], max_count)
            balanced.extend([(label, d) for d in filled_data])
        return balanced

    # 上采样并加入一位扰动
    def generate_disturbed_data(self, data, max_num):
        maxs, mins = self.get_column_distribution(data)
        col_num = len(maxs)
        for i in range(max_num - len(data)):
            pos = random.randint(0, len(data) - 1)
            new_item = data[pos]
            col_to_modify = random.randrange(0, col_num, step=1)
            new_item[col_to_modify] = float(random.randint(mins[col_to_modify], maxs[col_to_modify]))
            data.append(new_item)
        return data

    def get_column_distribution(self, data):
        columns = list(zip(*data))
        maxs = []
        mins = []
        for col in columns:
            col = set(col)
            # -8 是一个特殊值
            if -8 in col:
                col.remove(-8)
            maxs.append(max(col))
            mins.append(min(col))
        return maxs, mins

    def load_data(self):
        cache_path = self.data_dir + self.cache_name
        if not os.path.exists(cache_path):
            self.cache_data()
        with open(cache_path, 'r', encoding='utf-8') as fin:
            print('Loading cached dataset from {}'.format(cache_path))
            data = json.load(fin)
            print('Loaded {} lines'.format(len(data)))
            data = utils.split_avg(data, self.n_fold)
        random.shuffle(data)
        return data

    def cache_data(self, get_null_cols=False):
        if not get_null_cols:
            print('Caching tensorized data...')

        fin = open(self.data_dir + 'train_data.csv', 'r')
        data = pds.read_csv(fin)
        happiness = data['happiness'].tolist()

        null_stat = data.isnull().sum().to_dict()
        null_name_list = list(filter(lambda x: null_stat[x] > 0, null_stat))
        null_name_list.extend(['id', 'happiness'])
        null_name_list = list(set(null_name_list))
        if get_null_cols:
            fin.close()
            return null_name_list
        data_no_null = data.drop(null_name_list, axis=1)

        data_cache = []
        cache_path = self.data_dir + self.cache_name
        cached_num = 0

        for index, row in data_no_null.iterrows():
            data_cache.append((happiness[index], row.tolist()))
            cached_num += 1

        random.shuffle(data_cache)
        filled_num = len(data_cache) - cached_num
        fout = open(cache_path, 'w', encoding='utf-8')
        json.dump(data_cache, fout, indent=None, separators=(',', ':'))
        fin.close()
        fout.close()

        print('{} lines raw data loaded'.format(cached_num))
        print('{} lines generated data filled'.format(filled_num))
        print('{} lines data cached'.format(cached_num + filled_num))

    def load_submit(self):
        cache_path = self.data_dir + self.submit_name
        if not os.path.exists(cache_path):
            self.cache_submit()
        with open(cache_path, 'r', encoding='utf-8') as fin:
            print('Loading submit dataset from {}'.format(cache_path))
            data = json.load(fin)
            print('Done')
        return data

    def cache_submit(self):
        fin = open(self.data_dir + 'test_data.csv', 'r')
        data = pds.read_csv(fin)
        ids = data['id'].tolist()

        if self.null_column_names is None:
            self.null_column_names = self.cache_data(get_null_cols=True)
            self.null_column_names.remove('happiness')
        data_no_null = data.drop(self.null_column_names, axis=1)

        data_cache = []
        cache_path = self.data_dir + self.submit_name

        for index, row in data_no_null.iterrows():
            data_cache.append((ids[index], row.tolist()))

        fout = open(cache_path, 'w', encoding='utf-8')
        json.dump(data_cache, fout, indent=None, separators=(',', ':'))

        fin.close()
        fout.close()

    def generate_train_data(self):
        random.shuffle(self.train_data)
        for i in range(0, len(self.train_data), self.batch_size):
            pos = i + self.batch_size
            if pos > len(self.train_data):
                batch_size = len(self.train_data) - i
            else:
                batch_size = self.batch_size
            if batch_size == 1:
                continue
            td = self.train_data[i: i + self.batch_size]
            max_len = max([len(x[1]) for x in td])
            examples = np.zeros([batch_size, max_len], dtype=np.float32)
            labels = np.zeros([batch_size], dtype=np.float32)
            for j, e in enumerate(td):
                examples[j][:len(e[1])] = e[1]
                labels[j] = e[0]
            if self.conv:
                examples = np.expand_dims(examples, -1)
            yield examples, np.log1p(labels)

    def generate_test_data(self):
        for i in range(0, len(self.test_data), self.batch_size):
            pos = i + self.batch_size
            if pos > len(self.test_data):
                batch_size = len(self.test_data) - i
            else:
                batch_size = self.batch_size
            if batch_size == 1:
                continue
            td = self.test_data[i: i + self.batch_size]
            max_len = max([len(x[1]) for x in td])
            examples = np.zeros([batch_size, max_len], dtype=np.float32)
            labels = np.zeros([batch_size], dtype=np.float32)
            for j, e in enumerate(td):
                examples[j][:len(e[1])] = e[1]
                labels[j] = e[0]
            if self.conv:
                examples = np.expand_dims(examples, -1)
            yield examples, np.log1p(labels)

    def generate_dev_data(self):
        for i in range(0, len(self.dev_data), self.batch_size):
            pos = i + self.batch_size
            if pos > len(self.dev_data):
                batch_size = len(self.dev_data) - i
            else:
                batch_size = self.batch_size
            if batch_size == 1:
                continue
            dd = self.dev_data[i: i + self.batch_size]
            max_len = max([len(x[1]) for x in dd])
            examples = np.zeros([batch_size, max_len], dtype=np.float32)
            labels = np.zeros([batch_size], dtype=np.float32)
            for j, e in enumerate(dd):
                examples[j][:len(e[1])] = e[1]
                labels[j] = e[0]
            if self.conv:
                examples = np.expand_dims(examples, -1)
            yield examples, np.log1p(labels)

    def generate_submit_data(self):
        if self.submit_data is None:
            self.submit_data = self.load_submit()
        for i in range(0, len(self.submit_data), self.batch_size):
            pos = i + self.batch_size
            if pos > len(self.submit_data):
                batch_size = len(self.submit_data) - i
            else:
                batch_size = self.batch_size
            if batch_size == 1:
                continue
            td = self.submit_data[i: i + self.batch_size]
            max_len = max([len(x[1]) for x in td])
            examples = np.zeros([batch_size, max_len], dtype=np.float32)
            ids = np.zeros([batch_size], dtype=np.int64)
            for j, e in enumerate(td):
                examples[j][:len(e[1])] = e[1]
                ids[j] = e[0]
            if self.conv:
                examples = np.expand_dims(examples, -1)
            yield examples, ids + 8001

    def generate_full_data(self):
        full_data = []
        for d in self.data:
            full_data.extend(d)
        random.shuffle(full_data)
        for i in range(0, len(full_data), self.batch_size):
            pos = i + self.batch_size
            if pos > len(full_data):
                batch_size = len(full_data) - i
            else:
                batch_size = self.batch_size
            if batch_size == 1:
                continue
            td = full_data[i: i + self.batch_size]
            max_len = max([len(x[1]) for x in td])
            examples = np.zeros([batch_size, max_len], dtype=np.float32)
            labels = np.zeros([batch_size], dtype=np.float32)
            for j, e in enumerate(td):
                examples[j][:len(e[1])] = e[1]
                labels[j] = e[0]
            if self.conv:
                examples = np.expand_dims(examples, -1)
            yield examples, np.log1p(labels)


if __name__ == '__main__':
    dg = DataGenerator('raw_data/', 10, conv=True)
    for example, ids in dg.generate_submit_data():
        print(example.shape)
        print(ids)
        break
    for example, ids in dg.generate_dev_data():
        print(example.shape)
        print(ids)
        break
