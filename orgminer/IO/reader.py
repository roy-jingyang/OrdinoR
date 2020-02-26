# -*- coding: utf-8 -*-

"""This module contains methods for importing event log data from files 
in specific formats.

Event log formats currently supported:

    - Disco-exported CSV format (https://fluxicon.com/disco/)
    - eXtensible Event Stream (XES) (http://xes-standard.org/)

All methods in this module should return the successfully imported event 
log as a pandas DataFrame.

Notes
-----
The original event log data should at least provide the information of 
the activity ids. It is expected that the resource ids and timestamps 
are presented as well (not mandatory though).
Other event data attributes can be appended according to different 
purposes of working projects and event data available.

See Also
--------
pandas.DataFrame : The primary pandas data structure.
"""
def _describe_event_log(el):
    """Output descriptive information for a successfully imported event
    log.

    Parameters
    ----------
    el : DataFrame
        An event log imported.

    Returns
    -------
    """
    print('-' * 80)
    print('Number of events:\t\t{}'.format(len(el)))
    print('Number of cases:\t\t{}'.format(len(el.groupby('case_id'))))
    #print('Event log attributes:\n\t')
    print('-' * 80)


def read_disco_csv(f, header=True):
    """Import an event log from a file in CSV (Column-Separated Values)
    format, exported from Disco.

    There are four expected "default" event attributes, including:

        - case_id,
        - activity,
        - resource,
        - timestamp,

    that start as the first four columns.

    Parameters
    ----------
    f : File object
        File object of the event log being imported.
    header : bool, optional, default True
        A boolean flag indicating whether the input event log file
        contains a header line. Defaults to ``True``, i.e., the provided 
        CSV file is expected to be having a header line.

    Returns
    -------
    el : DataFrame
        An event log.
    """
    from csv import reader

    is_header_line = header

    ld = list()
    line_count = 0

    # default attributes consistent with Disco export function
    attributes = {
        'case_id': 0,
        'activity': 1,
        'resource': 2,
        'timestamp': 3
    }
    num_default_attributes = len(attributes)
    has_additional_attributes = False
    attributes_registered = False

    for row in reader(f):
        if not row:
            continue

        if not attributes_registered:
            has_additional_attributes = len(row) > num_default_attributes
            if has_additional_attributes:
                # register them
                if is_header_line:
                    for i, name in enumerate(row[num_default_attributes:]):
                        col_index = i + num_default_attributes
                        attributes[name] = col_index
                else:
                    # no header line scanned but additional attributes exist
                    for i in range(len(row[num_default_attributes:])):
                        col_index = i + num_default_attributes
                        name = 'unknown_attr_col{}'.format(col_index + 1)
                        attributes[name] = col_index
            else:
                pass
            attributes_registered = True

        if is_header_line:
            is_header_line = False
        else:
            e = dict()
            for attr, col_index in attributes.items():
                e[attr] = row[col_index]
            ld.append(e)

        line_count += 1

    from pandas import DataFrame
    el = DataFrame(ld)

    print('Imported successfully. {} lines scanned.'.format(line_count))

    _describe_event_log(el)
    return el


def read_xes(f):
    """Import an event log from a file in IEEE XES (eXtensible Event 
    Stream) format.

    This is a wrapper method around the XES import feature provided in
    the PM4Py library. See: `<http://pm4py.org/>`_

    There are four expected default event attributes, including:

        - case_id (mapped from `case:concept:name`),
        - activity (mapped from `concept:name`),
        - resource (mapped from `org:resource`),
        - timestamp (mapped from `time:timestamp`).

    Parameters
    ----------
    f : File object
        File object of the event log being imported.

    Returns
    -------
    el : DataFrame
        An event log.
    """
    from pm4py.objects.log.importer.xes import factory
    pm4py_log = factory.import_log_from_string(f.read(), 
        parameters={'index_trace_indexes': True})
    from pm4py.objects.conversion.log.versions import to_dataframe
    df = to_dataframe.apply(pm4py_log).rename(columns={
        'case:concept:name': 'case_id',
        'concept:name': 'activity',
        'org:resource': 'resource',
        'time:timestamp': 'timestamp'
    })
    import re
    df['timestamp'] = df['timestamp'].apply(
        lambda x: x.strftime('%Y/%m/%d %H:%M:%S.%f'))
    
    _describe_event_log(df)
    return df

