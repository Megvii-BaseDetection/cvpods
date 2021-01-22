# Use Configs

cvpods treats config as a class.
In addition to the basic operations that access and update a config, we provide
the following extra functionalities:

1. The config can have `_BASE_` field, which will load a base config first.
   Values in the base config will be overwritten in sub-configs, if there are any conflicts.
   We provided several base configs under configs directroy for standard model architectures.
2. We provide config `find` and `diff` method, which will help you to find setting value and diff two config objects.  

### Use Configs

Some basic usage of the `Config` object is shown below:
```python
from config import config
config.find("weight")    # get all keys in config if it contains keyword "weight"
base_config = config.__class__.__base__()   # get base config object
config.xxx = yyy      # add new configs for your own custom components
config.diff(base_config)    # return a dict contains all difference between given config
config.show_diff(base_config)  # show difference
```


### Best Practice with Configs

1. Treat the configs you write as "code": avoid copying them or duplicating them; use "_BASE_"
   instead to share common parts between configs.

2. Keep the configs you write simple: don't include keys that do not affect the experimental setting.

3. Save a full config together with a trained model, and use it to run inference.
   This is more robust to changes that may happen to the config definition
   (e.g., if a default value changed).
