#!/usr/bin/env python3
"""
Mapping aggregation of (possibly) nested dictionaries
"""
from typing import Mapping, Union

class MappingAggregator:
    def _isnumber(value: any) -> bool:
        return isinstance(value, (int, float))

    def _add(d1: Mapping[str, any], d2: Mapping[str, any]):
        if MappingAggregator._isnumber(d1) and MappingAggregator._isnumber(d2):
            return d1 + d2
        
        output = {}
        for key in d1.keys():
            if key in d2.keys():
                output[key] = MappingAggregator._add(d1[key], d2[key])
            else:
                output[key] = d1[key]
        
        for key in d2.keys():
            if key not in output.keys():
                if key in d1.keys():
                    output[key] = MappingAggregator._add(d1[key], d2[key])
                else:
                    output[key] = d2[key]
        return output

    def add(d1: Mapping[str, any], d2: Mapping[str, any]):
        output = MappingAggregator._add(d1, d2)
        return output

    def _div(value, divisor):
        if MappingAggregator._isnumber(value):
            return value / divisor
        
        if not isinstance(value, dict):
            raise TypeError

        output = {}
        for key in value.keys():
            output[key] = MappingAggregator._div(value[key], divisor)
        return output

    def div(dict: Mapping[str, any], divisor: Union[float, int]):
        if not MappingAggregator._isnumber(divisor):
            raise TypeError
        
        return MappingAggregator._div(dict, divisor)