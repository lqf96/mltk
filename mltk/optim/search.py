from collections.abc import Iterator

from mltk import util

__all__ = [
    "SearchSpace",
    "RandomSearchIterator",
    "GridSearchIterator"
]

class SearchSpace(object):
    def __init__(self, search_settings, transform, transform_args=(), transform_kwargs={}):
        ## Random search settings
        self.search_settings = search_settings
        ## Parameter transform function
        self.transform = transform
        ## Arguments for parameter transform function
        self.transform_args = transform_args
        ## Keyword arguments for parameter transform function
        self.transform_kwargs = transform_kwargs
    def __call__(self, rand=util.default_rand):
        search_settings = self.search_settings
        transform = self.transform
        # Dictionary search setting
        if isinstance(search_settings, dict):
            # Sample parameter combination
            sample = next(RandomSearchIterator(search_settings, 1, rand))
            # Transform sample
            if transform is not None:
                sample = transform(*self.transform_args, **self.transform_kwargs, **sample)
        # Single or list search settings
        else:
            single_setting = False
            # Convert single setting to list
            if not isinstance(search_settings, list):
                search_settings = [search_settings]
                single_setting = True
            # Sample parameters
            sample = tuple((sampler(rand) for sampler in search_settings))
            # Transform sample
            if transform is not None:
                sample = transform(*self.transform_args, *sample, **self.transform_kwargs)
            elif single_setting:
                sample = sample[0]
        return sample

def _flatten_search_keys(search_settings):
    flatten_keys = []
    # Check and flatten keys
    for key in search_settings.keys():
        # Parameter search group
        if isinstance(key, tuple):
            # Check all sub-keys in parameters search group
            for sub_key in key:
                if not isinstance(sub_key, str):
                    raise ValueError("Illegal sub-key {} in parameter search group {}".format(
                        sub_key, key
                    ))
            flatten_keys += key
        # Single parameter
        elif isinstance(key, str):
            flatten_keys.append(key)
        # Illegal parameter for searching
        else:
            raise ValueError("Illegal parameter {} for grid search".format(key))
    return flatten_keys

class RandomSearchIterator(Iterator):
    def __init__(self, search_settings, n_searches, rand=util.default_rand):
        ## Random search settings
        self.search_settings = search_settings
        ## Number of searches
        self.n_searches = n_searches
        ## Random number generator
        self.rand = rand
        ## Flatten search setting keys
        self._flatten_keys = _flatten_search_keys(search_settings)
        ## Iterator count
        self._count = 0
    def __len__(self):
        return self.n_searches
    def __next__(self):
        try:
            return self.next()
        except Exception as e:
            if not isinstance(e, StopIteration):
                import traceback
                traceback.print_exc()
            raise e
    def next(self):
        # Check and update iterator count
        if self._count>=self.n_searches:
            raise StopIteration
        self._count += 1
        # Sample parameters
        params = []
        for key, sampler in self.search_settings.items():
            value = sampler(self.rand)
            # Parameter group
            if isinstance(key, tuple):
                params += value if isinstance(value, tuple) else [value]*len(key)
            # Single parameter
            else:
                params.append(value)
        return dict(zip(self._flatten_keys, params))

class GridSearchIterator(Iterator):
    def __init__(self, search_settings):
        ## Grid search settings
        self.search_settings = search_settings
        ## Search setting keys
        self._keys = list(search_settings.keys())
        ## Flatten search setting keys
        self._flatten_keys = _flatten_search_keys(search_settings)
        ## Parameter combinations iterator
        self._param_comb_iter = product(*search_settings.values())
    def __len__(self):
        n_combs = 1
        # Compute and return number of combinations
        for values in self.search_settings.values():
            n_combs *= len(values)
        return n_combs
    def __next__(self):
        params = []
        for key, value in zip(self._keys, next(self._param_comb_iter)):
            # Parameters group
            if isinstance(key, tuple):
                params += value if isinstance(value, tuple) else [value]*len(key)
            # Single parameter
            else:
                params.append(value)
        # Produce dictionary of parameter combination
        yield dict(zip(self._flatten_keys, params))
