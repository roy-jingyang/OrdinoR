import ordinor.constants as const

def _describe_event_log(el):
    print('-' * 80)
    print('Number of events:\t\t{}'.format(len(el)))
    print('Number of cases:\t\t{}'.format(len(el.groupby(const.CASE_ID))))
    print('-' * 80)
