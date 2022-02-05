def powerset_exclude_headtail(s, reverse=False, depth=None):
    """Python recipe: this function returns a power set of a given set
    of elements, but excluding the empty set and the given set itself,
    as a generator.

    Parameters
    ----------
    s : set or frozenset
        A given set of elements. 
    reverse : bool, optional, default True
        A boolean flag determining whether the generated power set (as a 
        generator) delivers sets with lower cardinality first or higher 
        ones. Defaults to ``True``, i.e. the lower ones before the 
        higher.
    depth : int, optional, default None
        The upper bound (or lower bound) of cardinality that filters the 
        sets to be generated. Defaults to ``None``, i.e. the whole power
        set will be returned.

    Returns
    -------
    generator
        A power set generated.
    """
    from itertools import chain, combinations
    s = list(s)
    if reverse:
        end = 0 if depth is None else (len(s) - 1 - depth)
        return (chain.from_iterable(combinations(s, r) 
            for r in range(len(s) - 1, end, -1)))
    else:
        end = len(s) if depth is None else (1 + depth)
        return (chain.from_iterable(combinations(s, r) 
            for r in range(1, end)))

def unique_k_partitions(s, k):
    """
    Return all possible partitions of a given set into k subsets, using
    the Knuth's algorithm U.
    This implementation is from a gist, copyright to GitHub user
    "olooney".
    
    Parameters
    ----------
    s : set or frozenset
        A given set of elements. 
    k : int
        The number of subsets (partitions). Must be a positive integer in
        [1, len(s)].
    
    Returns
    -------
    generator

    References
    ----------
    https://gist.github.com/olooney/8607faf1ee609b7c4da26f41f766a977
    """
    l = list(s)

    if len(l) < 2:
        return []

    def visit(n, a):
        ps = [set() for i in range(k)]
        for j in range(n):
            ps[a[j + 1]].add(l[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(l)
    a = [0] * (n + 1)
    for j in range(1, k + 1):
        a[n - k + j] = j - 1
    return f(k, n, 0, n, a)
