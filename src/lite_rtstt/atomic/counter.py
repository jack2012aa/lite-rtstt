import threading


class Counter:

    def __init__(self):
        self.__lock = threading.Lock()
        self.__condition = threading.Condition(self.__lock)
        self.__count = 0

    def increment(self):
        self.__lock.acquire()
        self.__count += 1
        self.__condition.notify_all()
        self.__lock.release()

    def wait_for(self, count: int):
        with self.__lock:
            while self.__count < count:
                self.__condition.wait()
