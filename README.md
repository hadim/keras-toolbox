# keras-toolbox

[![PyPI version](https://img.shields.io/pypi/v/keras-toolbox.svg?maxAge=2591000)](https://pypi.org/project/keras-toolbox/)
[![License ](https://img.shields.io/pypi/l/keras-toolbox.svg?maxAge=2591000)](LICENSE)

*Your every day Keras toolbox.*

The spirit of this library is to add some features missing from Keras and that make your workflow more smooth and easier. If these features start to be widely used it could be a good idea to propose them as a PR in the Keras repo.

Current features are :

- **`Monitor` callbacks** to easily get information about your model training. Current callbacks allow to write the state in a file as a JSON or to send the state to a Telegram user ID.

- **Visualization functions** : some functions to easily visualize weights and feature maps form a specific layer or all the layers.

- **Augmentation functions** : trying to replace the Keras augmentation API but the code is very crappy at the moment.

## Installation

```
pip install keras-toolbox
```

## Dependencies

- Python 3 only (it's time to move)
- the classics `numpy`, `pandas`, `scipy`
- `Keras`
- `matplotlib` for the visualization
- `python-telegram-bot` for the `TelegramMonitor` callback

## Documentation

The project is not big enough to write a proper doc for now. See this [Notebook](doc/example.ipynb) for a short documentation.

## Author

- Hadrien Mary <hadrien.mary@gmail.com>

## License

MIT license. See [LICENSE](LICENSE).
