def f1_score(rl, om, fitness, precision):
    """
    Calculate the F1-score combining fitness and precision measures. 

    Parameters
    ----------
    rl : pandas.DataFrame
        A resource log.
    om : OrganizationalModel
        An organizational model.
    fitness : function
        Function for calculating fitness.
    precision : function
        Function for calculating precision.

    Returns
    -------
    float
        The resulting f1-score.
    
    See Also
    --------
    ordinor.conformance.fitness
    ordinor.conformance.precision
    """
    fitness_score = fitness(rl, om)
    precision_score = precision(rl, om)
    return (
        (2 * fitness_score * precision_score) 
        / 
        (fitness_score + precision_score)
    )
