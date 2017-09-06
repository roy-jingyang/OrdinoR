#! /usr/bin/bash

python3 src/mine.py data/2015/data2015/csv\ formats/BPIC15_5.csv output/bpic2015_m5_HoW 'Handover.ICCDCM'
python3 src/mine.py data/2015/data2015/csv\ formats/BPIC15_5.csv output/bpic2015_m5_Jcases 'WorkingTogether.SAR'
python3 src/mine.py data/2015/data2015/csv\ formats/BPIC15_5.csv output/bpic2015_m5_Jacts 'SimilarTask.ED'

