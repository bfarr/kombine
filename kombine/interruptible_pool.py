"""
This is a drop-in replacement for multiprocessing's pool that
plays better with keyboard interrupts.  This implimentation is a modified
version of one originally written by Peter K. G. Williams <peter@newton.cx>
for emcee:

    * `<https://github.com/dfm/emcee/blob/master/emcee/interruptible_pool.py>`_

which was an adaptation of a method written by John Reese, shared as

    * `<https://github.com/jreese/multiprocessing-keyboardinterrupt/>`_
"""

import signal
import functools
from multiprocessing.pool import Pool as MPPool
from multiprocessing import TimeoutError


def _initializer_wrapper(initializer, *args):
    """
    Ignore SIGINT. During typical keyboard interrupts, the parent does the
    killing.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if initializer is not None:
        initializer(*args)

class Pool(MPPool):
    """
    A modified :class:`multiprocessing.pool.Pool` that handles :exc:`KeyboardInterrupts` in the
    :func:`map` method more gracefully.

    :param processes: (optional)
        The number of processes to use (defaults to number of CPUs).

    :param initializer: (optional)
        A callable to be called by each process when it starts.

    :param initargs: (optional)
        Arguments for *initializer*; called as ``initializer(*initargs)``.

    :param kwargs: (optional)
        Extra arguments. Python 2.7 supports a `maxtasksperchild` parameter.
    """
    def __init__(self, processes=None, initializer=None, initargs=(), **kwargs):
        self._wait_timeout = 3600

        new_initializer = functools.partial(_initializer_wrapper, initializer)
        super(Pool, self).__init__(processes, new_initializer, initargs, **kwargs)

    def map(self, func, items, chunksize=None):
        """
        A replacement for :func:`map` that handles :exc:`KeyboardInterrupt`.

        :param func:
            Function to apply to the items.

        :param items:
            Iterable of items to have `func` applied to.
        """
        # Call r.get() with a timeout, since a Condition.wait() swallows
        # KeyboardInterrupts without a timeout
        r = self.map_async(func, items, chunksize)

        while True:
            try:
                return r.get(self._wait_timeout)
            except TimeoutError:
                pass
            except KeyboardInterrupt:
                self.terminate()
                self.join()
                raise
