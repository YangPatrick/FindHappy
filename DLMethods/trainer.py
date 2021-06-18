import os
import time

import numpy as np
import torch
import torch.nn as nn

import src.model as md
import src.utils as utils
from src.data_generator import DataGenerator


class Trainer:
    def __init__(self, device, config):

        self.device = device
        self.name = config['name']
        self.conf = config['conf']
        self.data_dir = self.conf['data_dir']
        self.log_path = self.conf['log_dir'] + self.name + '.log'
        self.model_path = self.conf['model_dir'] + self.name + '.ptm'
        self.batch_size = self.conf['batch_size']
        self.learning_rate = self.conf['learning_rate']
        self.decay_rate = self.conf['decay_rate']
        self.decay_freq = self.conf['decay_freq.']
        self.max_hanging_epoch = self.conf['max_hanging_epoch']
        self.model_selector = self.conf['model_selector']

        self.resnet_shrink = self.conf['resnet_shrink']
        self.resnet_depth = self.conf['resnet_depth']

        self.hidden_layers = self.conf['hidden_layers']
        self.hidden_size = self.conf['hidden_size']
        self.batch_norm = self.conf['batch_norm']

        print('ResNet Shrink: {}'.format(self.resnet_shrink))
        print('ResNet Depth: {}'.format(self.resnet_depth))

        print('Batch size: {}'.format(self.batch_size))
        print('Initial lr: {}'.format(self.learning_rate))
        print('Decay frequency: {}'.format(self.decay_freq))

        if self.model_selector == 'resnet':
            self.data_generator = DataGenerator(self.data_dir, self.batch_size, conv=True)
            self.model = md.ResNetClassifier(self.data_generator.channels, self.resnet_shrink, self.resnet_depth, num_classes=1).to(self.device)
        elif self.model_selector == 'mlp':
            self.data_generator = DataGenerator(self.data_dir, self.batch_size, conv=False)
            self.model = md.MLP(self.data_generator.emb_size, self.hidden_layers, self.hidden_size, self.batch_norm, num_classes=1).to(self.device)
        else:
            print('Model not implemented')
            exit(-1)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.decay_freq, gamma=self.decay_rate)

        open(self.log_path, 'w', encoding='utf-8').close()

    def write_log(self, epoch, loss, dev_loss, test_loss, dev_mse, test_mse, eps):
        with open(self.log_path, 'a', encoding='utf-8') as log:
            log_line = '{}-{}'.format(self.data_generator.test_pos, epoch)
            log_line += ' ' + 'loss={:.4f}'.format(loss)
            log_line += ' ' + 'dev_loss={:.4f}'.format(dev_loss)
            log_line += ' ' + 'test_loss={:.4f}'.format(test_loss)
            log_line += ' ' + 'dev_mse={:.6f}'.format(dev_mse)
            log_line += ' ' + 'test_mse={:.6f}'.format(test_mse)
            log_line += ' ' + 'eps={:.2f}'.format(eps)
            log.write(log_line + '\n')

    def save_model(self):
        torch.save({'model': self.model.state_dict()}, self.model_path)
        print('Model saved in {}'.format(self.model_path))

    def load_model(self):
        if os.access(self.model_path, os.R_OK):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            print('Model recovered from {}'.format(self.model_path))
        else:
            print('Model training from scratch')

    def train(self):
        min_mse = 10.0
        mse_list = []
        accu_loss = 0.0
        steps = 0
        epochs = 0
        hanging = 0
        self.model.train()
        start_time = time.time()
        while not self.data_generator.round_end:
            for examples, labels in self.data_generator.generate_train_data():
                examples = torch.from_numpy(examples).to(self.device)
                labels = torch.from_numpy(labels).to(self.device)
                output = self.model(examples).squeeze(1)
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                accu_loss += loss.detach().item()
                steps += 1
                if steps % 10 == 0:
                    print('\r{}'.format(steps), end='')
            epochs += 1
            time_cost = time.time() - start_time
            steps_per_sec = steps / time_cost
            average_loss = accu_loss / steps
            dev_loss, dev_mse = self.validate()
            test_loss, test_mse = self.evaluate()
            print('\r[{}] train_loss={:.6f} dev_loss={:.6f} test_loss={:.6f} bps={:.1f}'.format(epochs, average_loss, dev_loss, test_loss, steps_per_sec))
            print('dev_mse={:.6f} test_mse={:.6f}'.format(dev_mse, test_mse))
            print('Current lr is: {}'.format(self.optimizer.param_groups[0]['lr']))
            print()
            self.write_log(epochs, average_loss, dev_loss, test_loss, dev_mse, test_mse, steps_per_sec * self.batch_size)
            if test_mse < min_mse:
                min_mse = test_mse
                hanging = 0
                # self.save_model()
            else:
                hanging += 1
            if hanging == self.max_hanging_epoch:
                # self.predict()
                # exit(1)
                mse_list.append(min_mse)
                min_mse = 10.0
                epochs = 0
                hanging = 0
                if not self.data_generator.round_end:
                    print('\nCurrent mse: ' + str([float('{:.6f}'.format(i)) for i in mse_list]) + '\n')
                    print('------------Fold-{}------------\n'.format(self.data_generator.test_pos))
                else:
                    print('\n------Result after {} fold------'.format(self.data_generator.n_fold))
                self.data_generator.switch_train_test()
                self.reset()
            accu_loss = 0.0
            steps = 0
            start_time = time.time()
        print('Average mse={:.6f}'.format(sum(mse_list) / len(mse_list)))

    def full_train(self):
        end = False
        accu_loss = 0.0
        min_loss = 10.0
        steps = 0
        epochs = 0
        hanging = 0
        self.model.train()
        while not end:
            for examples, labels in self.data_generator.generate_full_data():
                examples = torch.from_numpy(examples).to(self.device)
                labels = torch.from_numpy(labels).to(self.device)
                output = self.model(examples).squeeze(1)
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                accu_loss += loss.detach().item()
                steps += 1
                if steps % 10 == 0:
                    print('\r{}'.format(steps), end='')
            epochs += 1
            avg_loss = accu_loss / steps
            print('\r[{}] train_loss={:.6f}'.format(epochs, avg_loss))
            print('Current lr is: {}'.format(self.optimizer.param_groups[0]['lr']))
            print()
            if avg_loss < min_loss:
                min_loss = avg_loss
                hanging = 0
                self.save_model()
            else:
                hanging += 1
            if hanging == self.max_hanging_epoch:
                self.reset()
                self.load_model()
                self.predict()
                exit(0)
            accu_loss = 0.0
            steps = 0

    def evaluate(self):
        self.model.eval()
        mse = 0.0
        accu_loss = 0.0
        steps = 0
        for examples, labels in self.data_generator.generate_test_data():
            examples = torch.from_numpy(examples).to(self.device)
            labels_tensor = torch.from_numpy(labels).to(self.device)
            output = self.model(examples)
            accu_loss += self.criterion(output.squeeze(1), labels_tensor).detach().item()
            result = output.squeeze(1).cpu().detach().numpy()
            mse += utils.square_error(np.expm1(labels), np.expm1(result))
            steps += 1
        avg_loss = accu_loss / steps
        mse = mse / (self.data_generator.test_num())
        self.model.train()
        return avg_loss, mse

    def validate(self):
        self.model.eval()
        mse = 0.0
        accu_loss = 0.0
        steps = 0
        for examples, labels in self.data_generator.generate_dev_data():
            examples = torch.from_numpy(examples).to(self.device)
            labels_tensor = torch.from_numpy(labels).to(self.device)
            output = self.model(examples)
            accu_loss += self.criterion(output.squeeze(1), labels_tensor).detach().item()
            result = output.squeeze(1).cpu().detach().numpy()
            mse += utils.square_error(np.expm1(labels), np.expm1(result))
            steps += 1
        avg_loss = accu_loss / steps
        mse = mse / (self.data_generator.dev_num())
        self.model.train()
        return avg_loss, mse

    def predict(self):
        self.load_model()
        self.model.eval()
        id_list = []
        happy_list = []
        for examples, ids in self.data_generator.generate_submit_data():
            examples = torch.from_numpy(examples).to(self.device)
            output = self.model(examples).squeeze(1).cpu().detach().numpy()
            result = np.expm1(output)
            id_list.extend(ids)
            happy_list.extend(result)
        self.model.train()
        utils.save_as_csv([id_list, happy_list], ['id', 'happiness'], self.data_dir + 'submit.csv')

    def reset(self):
        del self.model
        del self.criterion
        del self.optimizer
        del self.scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.model_selector == 'resnet':
            self.model = md.ResNetClassifier(self.data_generator.channels, self.resnet_shrink, self.resnet_depth, num_classes=1).to(self.device)
        elif self.model_selector == 'mlp':
            self.model = md.MLP(self.data_generator.emb_size, self.hidden_layers, self.hidden_size, self.batch_norm, num_classes=1).to(self.device)
        else:
            print('Model not implemented')
            exit(-1)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.decay_freq, gamma=self.decay_rate)
