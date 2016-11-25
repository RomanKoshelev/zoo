from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = int(buffer_size)
        self.num_experiences = 0
        self.buffer = deque()

    def __str__(self):
        return "%s:\n\t%s\n\t%s" % (
            self.__class__.__name__,
            "buffer_size: " + str(self.buffer_size),
            "num_experiences: " + str(self.num_experiences),
        )

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
