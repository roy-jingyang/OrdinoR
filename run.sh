#! /bin/bash

#python3 src/BPIC-log-extraction/BPIC-2012/merge-2012.py input/2012/data2012/filter_list.txt input/2012/data2012/FILTERED.csv output/merged-2012.csv
#mv output/merged-2012.csv input/
python3 src/BPIC-log-extraction/BPIC-2012/extract-2012.py input/merged-2012.csv 'Subcontracting.CCCDCM'
python3 src/BPIC-log-extraction/BPIC-2012/extract-2012.py input/merged-2012.csv 'Handover.CCCDCM'
python3 src/BPIC-log-extraction/BPIC-2012/extract-2012.py input/merged-2012.csv 'WorkingTogether.SAR'
python3 src/BPIC-log-extraction/BPIC-2012/extract-2012.py input/merged-2012.csv 'SimilarTask.ED'

