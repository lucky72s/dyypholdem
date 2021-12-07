import datetime


class Timer(object):

    def __init__(self, logger):
        self._start: datetime.datetime = None
        self._stop: datetime.datetime = None
        self._runtime: datetime.datetime = None
        self._split_start: datetime.datetime = None
        self._split_stop: datetime.datetime = None
        self._split_time: datetime.datetime = None
        self._elapsed: datetime.datetime = None
        self.logger = logger

    def start(self, message=None, log_level="TIMING"):
        if message is not None:
            self.logger.log(log_level, message)
        self._start = datetime.datetime.now()
        return self

    def stop(self, message=None, log_level="TIMING"):
        self._stop = datetime.datetime.now()
        self._runtime = self._stop - self._start
        if message is not None:
            self.logger.log(log_level, f"{message}: {self._runtime}")
        return self._runtime

    def time(self, message="Elapsed: "):
        self._elapsed = datetime.datetime.now() - self._start
        return message + str(self._elapsed)

    def split_start(self, message=None, log_level="TIMING"):
        self._split_start = datetime.datetime.now()
        if message is not None:
            self.logger.log(log_level, f"{message}...")
        return self._split_start

    def split_stop(self, message=None, log_level="TIMING"):
        self._split_stop = datetime.datetime.now()
        self._split_time = self._split_stop - self._split_start
        if message is not None:
            self.logger.log(log_level, f"{message}: {self._split_time}")
        return self._split_time

    def split_time(self, message="Elapsed in split: "):
        self._elapsed = datetime.datetime.now() - self._split_start
        return message + str(self._elapsed)
