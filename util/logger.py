import os, tensorboardX, time, sys, logging, json, numpy as np
class Logger:
    def __init__(self, log_dir, log_name='log', log_level=logging.INFO, log_console=False, log_file=True, log_tensorboard=True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_name = log_name
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if log_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)
        if log_file:
            file_handler = logging.FileHandler(os.path.join(log_dir, log_name + '.log'))
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

        self.log_tensorboard = log_tensorboard
        if log_tensorboard:
            self.tensorboard_writer = tensorboardX.SummaryWriter(os.path.join(log_dir, 'tensorboard'))

    def log_epoch(self, epoch, train_loss, train_acc, valid_acc, test_acc,train_f1=None, valid_f1=None, test_f1=None,train_recall=None, valid_recall=None, test_recall=None, train_auc = None, valid_auc = None, test_auc = None, val_time=None, test_time=None):
        if self.log_tensorboard:
            self.tensorboard_writer.add_scalar('train/loss', train_loss, epoch)
            self.tensorboard_writer.add_scalar('train/accuracy', train_acc, epoch)
            self.tensorboard_writer.add_scalar('valid/accuracy', valid_acc, epoch)
            self.tensorboard_writer.add_scalar('test/accuracy', test_acc, epoch)
            if train_f1:
                self.tensorboard_writer.add_scalar('train/f1', train_f1, epoch)
                self.tensorboard_writer.add_scalar('valid/f1', valid_f1, epoch)
                self.tensorboard_writer.add_scalar('test/f1', test_f1, epoch)
            if train_recall:
                self.tensorboard_writer.add_scalar('train/recall', train_recall, epoch)
                self.tensorboard_writer.add_scalar('valid/recall', valid_recall, epoch)
                self.tensorboard_writer.add_scalar('test/recall', test_recall, epoch)
            if train_auc:
                self.tensorboard_writer.add_scalar('train/auc', train_auc, epoch)
                self.tensorboard_writer.add_scalar('valid/auc', valid_auc, epoch)
                self.tensorboard_writer.add_scalar('test/auc', test_auc, epoch)
            if val_time:
                self.tensorboard_writer.add_scalar('valid/time', val_time, epoch)
            if test_time:
                self.tensorboard_writer.add_scalar('test/time', test_time, epoch)
        
        self.log(f'Epoch {epoch}: train_loss={train_loss:.6f}, train_acc={train_acc:.6f}, valid_acc={valid_acc:.6f}, test_acc={test_acc:.6f},train_f1={train_f1}, valid_f1={valid_f1}, test_f1={test_f1},train_recall={train_recall}, valid_recall={valid_recall}, test_recall={test_recall},train_auc={train_auc}, valid_auc={valid_auc}, test_auc={test_auc}, val_time={val_time}, test_time={test_time}')

    def log_epoch_simple(self, epoch, train_loss):
        if self.log_tensorboard:
            self.tensorboard_writer.add_scalar('train/loss', train_loss, epoch)
        self.log(f'Epoch {epoch}: train_loss={train_loss:.6f}')

    def log_distillation(self, epoch, train_loss, lr=None, teacher_loss=None, student_loss=None, best_loss=None):
        if self.log_tensorboard:
            self.tensorboard_writer.add_scalar('distillation/train_loss', train_loss, epoch)
            if lr is not None:
                self.tensorboard_writer.add_scalar('distillation/lr', lr, epoch)
            if teacher_loss is not None:
                self.tensorboard_writer.add_scalar('distillation/teacher_loss', teacher_loss, epoch)
            if student_loss is not None:
                self.tensorboard_writer.add_scalar('distillation/student_loss', student_loss, epoch)
            if best_loss is not None:
                self.tensorboard_writer.add_scalar('distillation/best_loss', best_loss, epoch)

        log_message = f'Epoch {epoch}: train_loss={train_loss:.6f}'
        if lr is not None:
            log_message += f', lr={lr:.6f}'
        if teacher_loss is not None:
            log_message += f', teacher_loss={teacher_loss:.6f}'
        if student_loss is not None:
            log_message += f', student_loss={student_loss:.6f}'
        if best_loss is not None:
            log_message += f', best_loss={best_loss:.6f}'
        self.log(log_message)

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)
    

    def __del__(self):
        if self.log_tensorboard:
            self.tensorboard_writer.close()
