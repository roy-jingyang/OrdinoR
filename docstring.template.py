# shebang

# This template is built following the instructions of the numpy doc 
# string.

# NOTE: Module
"""Module docstring summary.

Extended summary, citing [1]_.

See Also
--------

Notes
-----
The FFT is a fast implementation of the discrete Fourier transform:

.. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

The value of :math:`\omega` is larger than 5.


References
----------
.. [1] O. McNoleg, "The integration of GIS, remote sensing,
expert systems and adaptive co-kriging for environmental habitat
modelling of the Highland Haggis using object-oriented, fuzzy-logic
and neural-network techniques," Computers & Geosciences, vol. 22,
pp. 585-588, 1996.

Examples
--------
"""

from deprecated import deprecated

# NOTE: Function (deprecated)
@deprecated(version='1.2.1', reason='some particular reason')
def foo(x):
    """Function docstring summary.

    .. deprecated:: 1.6.0
      `ndobj_old` will be removed in NumPy 2.0.0, it is replaced by
      `ndobj_new` because the latter works also with array subclasses.

    Extended summary.


    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y
        Description of parameter `y` (with type not specified).

    z : int, optional

    order : {'C', 'F', 'A'}
        Description of `order`, with the default value appearing first.

    Returns
    -------
    int
        Description of anonymous integer return value.

    Raises
    ------
    LinAlgException
        If the matrix is not numerically invertible.

    See Also
    --------

    Notes
    -----

    References
    ----------

    Examples
    --------

    """
    pass


# NOTE: Class
class foo_class:
    """Class docstring summary.

    .. deprecated:: 1.6.0
      `ndobj_old` will be removed in NumPy 2.0.0, it is replaced by
      `ndobj_new` because the latter works also with array subclasses.

    Extended summary.


    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y
        Description of parameter `y` (with type not specified).

    z : int, optional

    order : {'C', 'F', 'A'}
        Description of `order`, with the default value appearing first.

    Attributes
    ----------
    attr_a : int
        Description of class attribute `attr_a`.

    Methods
    -------
    read_attr_a()
        Description of method `read_attr_a`.

    Raises
    ------
    LinAlgException
        If the matrix is not numerically invertible.

    See Also
    --------

    Notes
    -----

    References
    ----------

    Examples
    --------

    """

