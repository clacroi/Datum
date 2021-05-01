"""
Module implementing generic classes representing entries/observables attributes mappers,
i.e classes modifying entries/observables attributes.
"""

import inspect
from typing import List, Callable, Union, Tuple, Dict, Any, Optional

from datum.datasets import Entry, Observable
from datum.utils.exceptions import MissingAttributes


class EntryMapper():
    """Simple class to represent an entry mapper, i.e a function taking as input some
    attributes values from an entry (potentially a single attribute) and returning a
    set of values (potentially a single). For simplicity and easier declaration, there is no
    reference to classes Entry in EntryMapper class.

    Attributes:
        _in_attrs: ordered list of input attributes names to map. These attributes values
            may be passed to _func in the same order than _in_attrs, hence this order should be
            correct.
        _out_attrs: ordered list of attributes names outputted by _func on a set on input
            attributes values.
        _func: function to apply to a set of input attributes values to obtain output
            attributes values. _func output values are associated to _out_attrs attributes names
            in their order of "return", hence this order should be correct.
    """
    def __init__(self, in_attrs: List[str], out_attrs: List[str], func: Callable) -> None:
        func_arg_spec = inspect.getfullargspec(func)
        if len(func_arg_spec.args) != len(in_attrs):
            raise ValueError
        self._in_attrs: List[str] = in_attrs
        self._out_attrs: List[str] = out_attrs
        self._func: Callable = func

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        # TODO : serialize _func attribute
        return {'mapper_type': type(self).__name__,
                'in_attrs': self._in_attrs,
                'out_attrs': self._out_attrs}

    @property
    def in_attrs(self) -> List[str]:
        """Ordered list of input attributes names to map."""
        return self._in_attrs

    @property
    def out_attrs(self) -> List[str]:
        """Ordered list of mapped attributes names."""
        return self._out_attrs

    @property
    def func(self) -> Callable:
        """Function outputting out_attrs values from in_attrs values."""
        return self._func

    def apply_to(self, entry: Entry, apply_mode: str = 'force') -> Dict[str, Any]:
        """Apply mapper to input entry.

        Args:
            entry: input entry
            apply_mode: if 'optional' mapper is not applied if entry has not required attributes.

        Returns:
            dict containing mapper output (key, value) pairs
        """
        if apply_mode not in ['force', 'optional']:
            raise ValueError('apply_mode should be in |"force"|"optional"|')
        if not entry.check_attributes(self.in_attrs):
            if apply_mode == 'force':
                raise MissingAttributes(entry, self.in_attrs)
            return {}
        out = self([entry[attr] for attr in self.in_attrs])
        if len(self.out_attrs) > 1:
            return {attr: out[k] for k, attr in enumerate(self.out_attrs)}
        else:
            return {self.out_attrs[0]: out}

    def __call__(self, values: Union[List, Tuple, Dict[str, Any]]) -> Any:
        """Apply mapper to set of input attributes values."""
        if isinstance(values, list) or isinstance(values, tuple):
            # TODO : check if len(values) and _func.func_code.co_argcount match ?
            return self._func(*values)
        elif isinstance(values, dict):
            # TODO : check if values keys and _func.func_code.co_varnames match ?
            return self._func(**values)
        else:
            raise Exception('Mapper input should be list, tuple or dict')

    def __repr__(self) -> str:
        """Return string representing self."""
        return '{} : {} -> {}'.format(self._func.__name__, self._in_attrs, self._out_attrs)


class ObservableMapper(EntryMapper):
    """Simple class to represent an observable attributes mapper. The only difference with parent
    class EntryMapper is that this type of mapper should apply to observable attributes,
    optionnally extended with some of its corresponding entry attributes (Ex : scaling coordinates
    to unit of an 'object' observable from its image (entry) dimensions) and potentially to a
    unique type of observable (_obs_type).

    Attributes:
        _obs_type: observable type the mapper should apply to. 'all' if it should
        _obs_in_attrs: list of observable input attributes
        _entry_in_attrs: list of observable corresponding entry input attributes
    """
    def __init__(self, obs_in_attrs: List[str], out_attrs: List[Any],
                 func: Callable, obs_type: Optional[str] = 'all',
                 entry_in_attrs: Optional[List[str]] = None):
        self._obs_in_attrs: List[str] = obs_in_attrs
        if entry_in_attrs:
            if set(obs_in_attrs) & set(entry_in_attrs):
                raise ValueError('Input observable attributes and input entry attributes '
                                 'should have different names. Otherwise it could result '
                                 'in an unexpected behavior at mapper execution.')
            self._entry_in_attrs: List[str] = entry_in_attrs
        else:
            self._entry_in_attrs: List[str] = []
        super().__init__(self.obs_in_attrs + self.entry_in_attrs, out_attrs, func)
        self._obs_type: str = obs_type

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        # TODO : serialize _func attribute
        return {'mapper_type': type(self).__name__,
                'obs_in_attrs': self._obs_in_attrs,
                'entry_in_attrs': self._obs_in_attrs,
                'in_attrs': self._in_attrs,
                'out_attrs': self._out_attrs}

    @property
    def obs_type(self) -> str:
        """Observable type the mapper should be applied on."""
        return self._obs_type

    @property
    def obs_in_attrs(self) -> List[str]:
        """Ordered list of observable input attributes names to map."""
        return self._obs_in_attrs

    @property
    def entry_in_attrs(self) -> List[str]:
        """Optional ordered list of corresponding entry input attributes names to map."""
        return self._entry_in_attrs

    def apply_to(self, observable: Observable,
                 entry: Optional[Entry] = None,
                 apply_mode: str = 'force') -> Dict[str, Any]:
        """Apply mapper to input observable and optional corresponding entry.

        Args:
            observable: input entry.
            entry: optional observable corresponding entry if mapper also takes as input
                entry attributes.
            apply_mode: if 'optional' mapper is not applied if entry has not required attributes.

        Returns:
            dict containing mapper output (key, value) pairs
        """
        if self.entry_in_attrs and not entry:
            raise ValueError('Mapper takes as input observable corresponding entry attributes.')
        if apply_mode not in ['force', 'optional']:
            raise ValueError('apply_mode should be in |"force"|"optional"|')
        if self.obs_type != 'all' and self.obs_type != observable.typ:
            raise ValueError('Observable mapper should apply to observables with type {} (vs. {})'
                             .format(self.obs_type, observable.typ))

        # Check that input observable has correct attributes
        if not observable.check_attributes(self.obs_in_attrs):
            if apply_mode == 'force':
                raise MissingAttributes(observable, self.obs_in_attrs)
            return {}
        # Check that input entry has correct attributes
        if self.entry_in_attrs:
            if not entry.check_attributes(self.entry_in_attrs):
                if apply_mode == 'force':
                    raise MissingAttributes(entry, self.entry_in_attrs)
                return {}
        out = self([observable[attr] for attr in self.obs_in_attrs] +
                   [entry[attr] for attr in self.entry_in_attrs])
        if len(self.out_attrs) > 1:
            return {attr: out[k] for k, attr in enumerate(self.out_attrs)}
        else:
            return {self.out_attrs[0]: out}
