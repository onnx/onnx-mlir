#!/usr/bin/env python3

import fasteners
import logging
import sys

logging.basicConfig(
    level = logging.INFO, format = '[%(asctime)s] %(levelname)s: %(message)s')

# sys.argv[1]: read-write lock
# sys.argv[2]: acquire or release
# sys.argv[3]: optional log message

def main():
    rwlock = fasteners.InterProcessReaderWriterLock(sys.argv[1])

    if sys.argv[2] == 'acquire':
        logging.info(sys.argv[3]) if sys.argv[3] else None
        rwlock.acquire_write_lock()
    elif sys.argv[2] == 'release':
        rwlock.release_write_lock()
        logging.info(sys.argv[3]) if sys.argv[3] else None
    else:
        logging.warning('{} unknown, no rwlock acquired/released'.format(sys.argv[2]))

if __name__ == "__main__":
    main()
