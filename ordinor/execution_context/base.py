from collections import defaultdict
import pickle

from pandas import DataFrame

from ordinor.utils.validation import check_convert_input_log
import ordinor.exceptions as exc
import ordinor.constants as const

class BaseMiner:
    """
    This class should be used as an interface base class for implementing any
    approach for learning execution contexts.

    Parameters
    ----------
    el : pandas.DataFrame, or pm4py EventLog
        An event log from which the execution contexts are learned.

    Attributes
    ----------
    is_ctypes_verified : bool
        A Boolean flag indicating whether the case types are verified.
    is_atypes_verified : bool
        A Boolean flag indicating whether the activity types are verified.
    is_ttypes_verified : bool
        A Boolean flag indicating whether the time types are verified.

    Methods
    -------
    derive_resource_log(el)
        Derive a resource log given the original event log, after the execution
        contexts have been discovered and verified (which is done the moment
        when an object is instantiated).
    get_type_by_value(value)
        Query the corresponding type given a value (of either an activity label,
        a case id, or a timestamp).
    get_values_by_type(type_name)
        Query the original values (of activity labels, case ids, or timestamps)
        given a type name.

    Notes
    -----
    The docstrings for this class and its methods provide the guidelines for any
    child class that implements a specific way of learning execution contexts by
    overwriting the following three methods:

        - `self._build_ctypes`
        - `self._build_atypes`
        - `self._build_ttypes`
    
    Any child class should inherit from class `BaseMiner` and enables the
    learning and storing of the following mappings:

    - from case ids to Case Types,

    .. math:: \\mathcal{C} \\rightarrow \\mathcal{CT}

    - from activity labels to Activity Types,

    .. math:: \\mathcal{A} \\rightarrow \\mathcal{AT}

    - from timestamps to Time Types,

    .. math:: \\mathcal{T} \\rightarrow \\mathcal{TT}

    These mappings should be built as python dicts of strings.

    Note that, any learning approach inherited from class `BaseMiner` must
    ensure that the derived types form a partitioning of the original values.

    For example, a learning approach must map each case captured in the event
    log to one case type and one case type only.
    """

    _ctypes = None
    is_ctypes_verified = False

    _atypes = None
    is_atypes_verified = False

    _ttypes = None
    is_ttypes_verified = False

    def __init__(self, el, **kwargs):
        """
        Instantiate an instance that implements a way of execution
        context learning.

        The constructor method should invoke the three methods for 
        building case types, activity types and time types, 
        respectively:

        - `self._build_ctypes`
        - `self._build_atypes`
        - `self._build_ttypes`

        The function signature may vary depending on the exact strategy 
        of learning execution contexts.

        Invoke method `self._verify()` to ensure that the built 
        types are valid, i.e., a set of such types would form a 
        partitioning of the original corresponding values.

        Note that any child class is required to have its own 
        constructor method written as there may be different 
        requirements for the inputs. Even if there is no additional
        requirements other than those in this base class, it is expected
        that constructor of the child class explicitly invokes this
        method.

        Parameters
        ----------
        el : pandas.DataFrame, or pm4py EventLog
            An event log from which the execution contexts are learned.

        Returns
        -------
        """
        if el is not None:
            el = check_convert_input_log(el)

        self._build_ctypes(el)
        self._build_atypes(el)
        self._build_ttypes(el)
        self._verify()


    def _build_ctypes(self, el, **kwargs):
        """
        Mine the case types.

        Each type should be stored as a key-value pair where the key is 
        of string type and the value is a set of strings.

        By default, this dimension is neglected, i.e., no case types are
        built, or you may consider all cases belonging to the same dummy
        type marked by an empty python string, ``''``.

        Parameters
        ----------
        el : pandas.DataFrame, or pm4py EventLog
            An event log from which the execution contexts are learned.

        Returns
        -------
        """
        self._ctypes = defaultdict(str) # defaults to ''
        self.is_ctypes_verified = True


    def _build_atypes(self, el, **kwargs):
        """
        Mine the activity types.

        Each type should be stored as a key-value pair where the key is 
        of string type and the value is a set of strings.

        By default, this dimension is neglected, i.e., no activity types
        are built, or you may consider all activity labels belonging to 
        the same dummy type marked by an empty python string, ``''``.

        Parameters
        ----------
        el : pandas.DataFrame, or pm4py EventLog
            An event log from which the execution contexts are learned.

        Returns
        -------
        """
        self._atypes = defaultdict(str) # defaults to ''
        self.is_atypes_verified = True


    def _build_ttypes(self, el, **kwargs):
        """
        Mine the time types.

        Each type should be stored as a key-value pair where the key is 
        of string type and the value is a set of strings.

        By default, this dimension is neglected, i.e., no time types
        are built, or you may consider all timestamps belonging to 
        the same dummy type marked by an empty python string, ``''``.

        Parameters
        ----------
        el : pandas.DataFrame, or pm4py EventLog
            An event log from which the execution contexts are learned.

        Returns
        -------
        """
        self._ttypes = defaultdict(str) # defaults to ''
        self.is_ttypes_verified = True


    def derive_resource_log(self, el):
        """
        Derive a resource log given the original log, after the 
        execution contexts have been discovered and verified (which is
        done the moment when an object is instantiated).

        Each resource event in the derived resource log is corresponded 
        with exactly an event in the source event log even when the 
        resource log is defined as a multiset.

        Note that, such resource events are required to contain resource
        information, i.e. events with no resource information in the 
        source event log will be discarded.

        Parameters
        ----------
        el : pandas.DataFrame, or pm4py EventLog
            An event log from which the execution contexts are learned.

        Returns
        -------
        pandas.DataFrame
            The derived resource log as a pandas DataFrame.
        """
        el = check_convert_input_log(el)

        rl = list()
        for event in el.to_dict(orient='records'):
            # NOTE: only events with resource information are considered
            if (event[const.RESOURCE] is not None 
                and event[const.RESOURCE] != ''):
                rl.append({
                    const.RESOURCE: event[const.RESOURCE],
                    const.CASE_TYPE: self._ctypes[event[const.CASE_ID]],
                    const.ACTIVITY_TYPE: self._atypes[event[const.ACTIVITY]],
                    const.TIME_TYPE: self._ttypes[event[const.TIMESTAMP]]
                })

        rl = DataFrame(rl)
        print('Resource Log derived: ', end='')
        print('{} events mapped onto {} execution contexts.\n'.format(
            len(el), 
            len(rl.drop_duplicates(
                subset=[const.CASE_TYPE, const.ACTIVITY_TYPE, const.TIME_TYPE],
                inplace=False))
        ))
        return rl


    def to_file(self, f):
        """
        Export the constructed execution context miner to an external 
        file using Python pickle.
        
        Parameters
        ----------
        f : File object
            A destination file to be written.

        Returns
        -------
        """
        if (self.is_ctypes_verified and self.is_atypes_verified and
            self.is_ttypes_verified):
            obj = {
                '_ctypes': self._ctypes,
                'is_ctypes_verified': self.is_ctypes_verified,
                '_atypes': self._atypes,
                'is_atypes_verified': self.is_atypes_verified,
                '_ttypes': self._ttypes,
                'is_ttypes_verified': self.is_ttypes_verified
            }
            pickle.dump(obj, f)
        else:
            raise exc.InvalidModelError(
                'The specified types do not form the respective partitioning.'
            )


    @classmethod
    def from_file(cls, f):
        """
        Import a constructed execution context miner to an external file 
        using Python pickle.

        Parameters
        ----------
        f : File object
            A destination file to be written.

        Returns
        -------
        (Corresponding miner class object)
            The imported constructed execution context miner.
        """
        obj = pickle.load(f)
        if (obj['is_ctypes_verified'] and obj['is_atypes_verified'] and
            obj['is_ttypes_verified']):
            ret = cls(el=None)
            ret.is_ctypes_verified = ret.is_atypes_verified = \
                ret.is_ttypes_verified = True
            ret._ctypes = obj['_ctypes'] if len(obj['_ctypes']) > 0 else\
                defaultdict(str)
            ret._atypes = obj['_atypes'] if len(obj['_atypes']) > 0 else\
                defaultdict(str)
            ret._ttypes = obj['_ttypes'] if len(obj['_ttypes']) > 0 else\
                defaultdict(str)

            ret._verify()
            return ret
        else:
            raise exc.InvalidModelError(
                'The specified types do not form the respective partitioning.'
            )


    def _verify_partition(self, whole_set, partitioning):
        """
        A helper function that is used for verifying if the keys in a 
        given partitioning (as a dict) is indeed a partitioning of 
        values in a given set.

        Parameters
        ----------
        whole_set : set

        partitioning : dict

        Returns
        -------
        Boolean
            The result of verification.
        """
        is_disjoint = True # given the internal data structure as a dict
        is_union = set(partitioning.keys()) == whole_set
        return is_disjoint and is_union


    def _verify(self):
        """
        Verify if the built execution contexts are valid and output
        prompting information.

        Parameters
        ----------

        Returns
        -------
        """
        if (self.is_ctypes_verified and self.is_atypes_verified and
                self.is_ttypes_verified):
            print('-' * 80)
            print('Count of Types in the current {}:'.format(
                self.__class__.__name__))
            n_ctypes = len(set(self._ctypes.values()))
            print('Number of C Types:\t\t{}'.format(
                n_ctypes if n_ctypes > 0 else 'n/a (1)'
            ))
            n_atypes = len(set(self._atypes.values()))
            print('Number of A Types:\t\t{}'.format(
                n_atypes if n_atypes > 0 else 'n/a (1)'
            ))
            n_ttypes = len(set(self._ttypes.values()))
            print('Number of T Types:\t\t{}'.format(
                n_ttypes if n_ttypes > 0 else 'n/a (1)'
            ))
            print('-' * 80)
        else:
            print('C Types:\t{}'.format(
                'VERIFIED' if self.is_ctypes_verified else 'INVALID'
            ))
            print('A Types:\t{}'.format(
                'VERIFIED' if self.is_atypes_verified else 'INVALID'
            ))
            print('T Types:\t{}'.format(
                'VERIFIED' if self.is_ttypes_verified else 'INVALID'
            ))
            raise exc.InvalidModelError(
                'The specified types do not form the respective partitioning'
            )


    # TODO:
    # from original activity labels/case ids/timestamps to types
    def get_type_by_value(self, value):
        """
        Query the built type given a value (of either an activity 
        label, a case id, or a timestamp).

        Parameters
        ----------
        value : str
            The given value to be queried.

        Returns
        -------
        type_name : str
            The corresponding type name.
        """
        raise NotImplementedError


    # from types to the originals
    def get_values_by_type(self, type_name):
        """
        Query the original values (of activity labels, case ids, or 
        timestamps) given a type name.

        Parameters
        ----------
        type_name : str
            The given type name to be queried.

        Returns
        -------
        values : list of str
            The corresponding values.
        """
        if type_name.startswith('CT'):
            return list(
                k for k, v in self._ctypes.items() if v == type_name
            )

        if type_name.startswith('AT'):
            return list(
                k for k, v in self._atypes.items() if v == type_name
            )

        if type_name.startswith('TT'):
            return list(
                k for k, v in self._ttypes.items() if v == type_name
            )

        raise exc.InvalidQueryError(
            f'Invalid value for parameter `type_name`: {type_name}'
        )
