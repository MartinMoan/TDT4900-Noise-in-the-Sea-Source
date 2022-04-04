#!/usr/bin/env python3
from typing import Iterable

class IndexMe:
    def __init__(self, values: Iterable[any]) -> None:
        self._indeces = list(range(0, len(values)))
        
        self._raw_values = values

    def _parse_slice_index(self, index):
        start, step, stop = index.start, index.step, index.stop
        
        if start is None:
            start = 0
        if step is None:
            step = 1
        if stop is None:
            stop = len(self)
            
        if type(start) != int:
            raise TypeError("slice index start is not int")
        if type(stop) != int:
            raise TypeError("slice index stop is not int")

        if type(step) != int:
            raise TypeError("slice index step is not int")
        
        return start, step, stop

    def __getitem__(self, index):
        if type(index) == slice:
            start, step, stop = self._parse_slice_index(index)
            
            output = []
            for i in range(start, stop, step):
                output.append(self[i])
            return output
        
        elif type(index) == int:
            raw_index = self._indeces[index]
            return self._raw_values[raw_index]
        
        raise TypeError("Index is not int nor slice")

    def __setitem__(self, index, value):
        if type(index) == slice:
            start, step, stop = self._parse_slice_index(index)
            for i in range(start, stop, step):
                self[i] = value
        elif type(index) == int:
            raw_index = self._indeces[index]
            raw_value = self._raw_values[index]
            
            return self._raw_values[raw_index]

    def __len__(self):
        return len(self._values)

    def __str__(self):
        return str(self._values)

if __name__ == "__main__":
    obj = IndexMe()

    print(obj[1:20])
    