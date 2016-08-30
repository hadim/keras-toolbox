import time

from keras.callbacks import Callback


class TelegramNotifier(Callback):
    """Very basic notifier that send message to Telegram during a Keras model training.
    For now I don't a lot of informations. Telegram allow a lot of different kind of informations to sent, from
    text to videos and images. The Telegram API allows the bot to ask question to the user.

    All of this make could make this callback very powerfull.

    Few random ideas :

    - at each epoch end returns some statistics about the loss and the accuracy.
    - set a parameter to control the whole verbosity
    - at the end of the training returns a plot of the loss against # of epoch
    """


    def __init__(self, api_token, chat_id):

        self.check_telegram_module()
        import telegram

        super(TelegramNotifier, self).__init__()

        self.bot = telegram.Bot(token=api_token)
        self.chat_id = chat_id

    def on_train_begin(self, logs={}):

        message = "Training started at {} for {} epoch with {} samples."
        self.notify(message.format(time.ctime(), self.params['nb_sample'], self.params['nb_epoch']))

    def on_batch_begin(self, batch, logs={}):

        pass

    def on_batch_end(self, batch, logs={}):

        pass

    def on_epoch_end(self, epoch, logs={}):

        message = "Epoch {}/{} is done at {}."
        self.notify(message.format(epoch+1, self.params['nb_epoch'], time.ctime()))

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

