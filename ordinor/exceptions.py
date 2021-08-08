"""
Warnings and errors used in OrdinoR
"""

import warnings

################### Errors ###################
class Error(Exception):
    """Base class for all errors defined in this module."""
    pass

class AlgorithmRuntimeError(Error, RuntimeError):
    """
    Exception raised when an algorithm encounters runtime error. 
    """
    def __init__(self, reason, suggestion=None):
        msg_sugg = '' if suggestion is None else suggestion
        self.message = f'{reason}. {msg_sugg}'

class FileParsingError(Error):
    """
    Exception raised when an importing file cannot be parsed.
    """
    def __init__(self, filepath, msg=None):
        spec_msg = '' if msg is None else msg
        self.message = f'''
            Cannot parse file {filepath}. {spec_msg}
        '''

class DataMissingError(Error):
    """
    Exception raised when required data cannot be found in an importing
    file. 
    """
    def __init__(self, msg=None):
        spec_msg = '' if msg is None else msg
        self.message = f'''
            Required data is missing in file. {spec_msg}
        '''

class InvalidModelError(Error, RuntimeError):
    """
    Exception raised when a model is invalid.
    """
    def __init__(self, msg=None):
        spec_msg = '' if msg is None else msg
        self.message = f'''
            Cannot verify the model. {spec_msg}
        '''

class InvalidQueryError(Error, ValueError):
    """
    Exception raised when attempting to query non-existent data.
    """
    def __init__(self, msg=None):
        spec_msg = '' if msg is None else msg
        self.message = f'''
            Unable to resolve the query. {spec_msg}
        '''

class InvalidParameterError(Error):
    """
    Exception raised when invalid parameter(s) is provided. 
    """
    def __init__(self, param, reason):
        self.message = f'''
            Found error with parameter `{param}`: {reason}
        '''

################### Warnings ###################
def warn_import_data_ignored(msg):
    """
    Warning issued when certain data is deliberately ignored in an
    importing file. 
    """
    warnings.warn(
        f'Data is ignored when importing file. {msg}', 
        UserWarning
    )

def warn_nan_returned(msg):
    """
    Warnings issued when attempting to calculate in an undefined
    scenario and thus obtaining NaN value(s).
    """
    warnings.warn(
        f'NaN value is returned. {msg}',
        RuntimeWarning
    )

def warn_runtime(msg):
    """
    Generic runtime warning. 
    """
    warnings.warn(msg, RuntimeWarning)

def warn_data_type_casted(old_dtype, new_dtype):
    """
    Warning issued when casting data types.
    """
    warnings.warn(
        f'{old_dtype} casted to {new_dtype}.',
        RuntimeWarning
    )
