# coding: utf-8

""" Utility class for a uniform Pool API for serial processing """

class SerialPool(object):

    def close(self):
        return

    def map(self, function, tasks, callback=None):
        results = []
        for task in tasks:
            result = function(task)
            if callback is not None:
                callback(result)
            results.append(result)
        return results
