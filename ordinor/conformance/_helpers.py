def f1_score(rl, om, fitness=None, precision=None):
    """
    Calculate the F1-score combining fitness and precision measures. 

    Parameters
    ----------
    rl : pandas.DataFrame
        A resource log.
    om : OrganizationalModel
        An organizational model.
    fitness : function, optional
        Function for calculating fitness. Defaults to
        `ordinor.conformance.fitness`.

    precision : function, optional
        Function for calculating precision. Defaults to
        `ordinor.conformance.precision`.

    Returns
    -------
    float
        The resulting f1-score.
    
    See Also
    --------
    ordinor.conformance.fitness
    ordinor.conformance.precision
    """
    if fitness is None:
        from ordinor.conformance import fitness
        fitness = fitness
    if precision is None:
        from ordinor.conformance import precision
        precision = precision

    fitness_score = fitness(rl, om)
    precision_score = precision(rl, om)
    return (
        (2 * fitness_score * precision_score) 
        / 
        (fitness_score + precision_score)
    )
