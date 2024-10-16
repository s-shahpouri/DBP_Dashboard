from pydicom.uid import UID
import datetime
import random

def generate_date_time_uid(prefix):
    
    uid = UID(f'{prefix}.{datetime.datetime.now():%Y%m%d%H%M%S}'
                                   f'{random.randrange(int(1e2), int(1e3))}.'
                                   f'{random.randrange(int(1e3), int(1e4))}.'
                                   f'{random.randrange(int(1e4), int(1e5))}') 
    return uid


