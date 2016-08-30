from datetime import datetime
import uuid
import json

import numpy as np

from keras.callbacks import Callback

from .utils import json_serial


class Monitor(Callback):
    """Monitor retrieve and continuously update the state a Keras model into the `self.state` attribute.
    """

    def __init__(self, date_format='%Y-%m-%d %H:%M'):

        super().__init__()

        self.date_format = date_format
        
        self.state = {}
        self.state['epochs'] = []
        
        # Add/generate these attributes in keras.models.Model ?
        self.state['name'] = 'A Keras model'
        self.state['model_id'] = str(uuid.uuid4())
        self.state['training_id'] = str(uuid.uuid4())

    def on_train_begin(self, logs={}):
        
        message = 'Monitor initialized.\n' \
                  'Name of the model is "{}"\n' \
                  'Model ID is {}\n' \
                  'Training ID is {}'
        self.notify(message.format(self.state['name'],
                                   self.state['model_id'],
                                   self.state['training_id']))

        # Add the model to the state
        self.state['model_json'] = self.model.to_json()
        
        self.state['params'] = self.params
        
        self.state['train_start_time'] = datetime.now()
        
        message = "Training started at {} for {} epochs with {} samples with a {} layers model."
        self.notify(message.format(self.state['train_start_time'].strftime(self.date_format),
                                   self.params['nb_epoch'],
                                   self.params['nb_sample'],
                                   len(self.model.layers)))
        
    def on_train_end(self, logs={}):
        
        self.state['train_end_time'] = datetime.now()
        self.state['train_duration'] = self.state['train_end_time'] - self.state['train_start_time']

        # In hours
        duration = (self.state['train_end_time'] - \
                    self.state['train_start_time']).total_seconds()
        self.state['train_duration'] = int(round(duration / 3600))
        
        message = "Training is done at {} for a duration of {} hours."
        self.notify(message.format(self.state['train_end_time'].strftime(self.date_format),
                                   self.state['train_duration']))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass
    
    def on_epoch_begin(self, epoch, logs={}):

        self.state['current_epoch'] = {}
        self.state['current_epoch']['epoch'] = epoch
        self.state['current_epoch']['start_time'] = datetime.now()

    def on_epoch_end(self, epoch, logs={}):

        self.state['current_epoch']['end_time'] = datetime.now()
        self.state['current_epoch']['logs'] = logs
        
        # In seconds
        duration = (self.state['current_epoch']['end_time'] - \
                    self.state['current_epoch']['start_time']).total_seconds()
        self.state['current_epoch']['duration'] = duration
        
        self.state['epochs'].append(self.state['current_epoch'])
        
        self.state['current_epoch']['average_minute_per_epoch'] = self.average_minute_per_epoch()
  
        message = "Epoch {}/{} is done at {}. Average minutes/epoch is {:.2f}."
        self.notify(message.format(epoch + 1, self.params['nb_epoch'],
                                   self.state['current_epoch']['end_time'].strftime(self.date_format),
                                   self.state['current_epoch']['average_minute_per_epoch']))\
        
        nice_logs = ' | '.join(["{} = {:.6f}".format(k, v) for k, v in logs.items()])
        self.notify("Logs are : {}".format(nice_logs))
        
        # Reset current epoch
        self.state['current_epoch'] = {}

    def notify(self, message, parse_mode=None):
        pass
    
    def average_minute_per_epoch(self):
        total_duration = self.state['current_epoch']['end_time'].total_seconds() - self.state['train_start_time'].total_seconds()
        total_duration /= 60
        minute_per_epoch = total_duration / (self.state['current_epoch']['epoch'] + 1)
        return minute_per_epoch


        
class PrintMonitor(Monitor):
    """This monitor only print messages with the classic `print` function
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def notify(self, message):
        print(message)
        
        
class TelegramMonitor(Monitor):
    """This monitor send messages to a Telegram chat ID with a bot.
    """
    
    def __init__(self, api_token, chat_id, **kwargs):
        
        self.check_telegram_module()
        import telegram
        
        super().__init__(**kwargs)
        
        self.bot = telegram.Bot(token=api_token)
        self.chat_id = chat_id
        
    def check_telegram_module(self):

        try:
            import telegram
        except ImportError:
            raise Exception("You don't have the python-telegram-bot library installed. "
                            "Please install it with `pip install python-telegram-bot -U` "
                            "in order to use the TelegramNotifier Keras callback.")

    def notify(self, message, parse_mode=None):
        
        ret = self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode=parse_mode)
        return ret
    
    
class FileMonitor(Monitor):
    """This monitor write a JSON file every time a message is sent. The JSON file contains all
        the state of the current training.
    """
    
    
    def __init__(self, filepath, **kwargs):
        
        super().__init__(**kwargs)
        
        self.filepath = filepath

    def notify(self, message, parse_mode=None):

        with open(self.filepath, 'w') as f:
            f.write(json.dumps(self.state, default=json_serial, indent=4, sort_keys=True))
    

