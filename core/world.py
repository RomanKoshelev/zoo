class World:
    __default__ = None

    def __init__(self):
        World.__default__ = self
        self.state = None

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *args):
        raise NotImplementedError

    def reset(self):
        self._reset()

    def _reset(self):
        raise NotImplementedError

    def step(self, action):
        return self._step(action)

    def _step(self, action):
        raise NotImplementedError

    def render(self):
        self._render()

    def _render(self):
        raise NotImplementedError

    @property
    def id(self):
        return self._get_id()

    def _get_id(self):
        raise NotImplementedError

    @property
    def obs_dim(self):
        return self._get_obs_dim()

    def _get_obs_dim(self):
        raise NotImplementedError

    @property
    def act_dim(self):
        return self._get_act_dim()

    def _get_act_dim(self):
        raise NotImplementedError

    @property
    def obs_box(self):
        return self._get_obs_box()

    def _get_obs_box(self):
        raise NotImplementedError

    @property
    def act_box(self):
        return self._get_act_box()

    def _get_act_box(self):
        raise NotImplementedError

    @property
    def summary(self):
        return self._get_summary()

    def _get_summary(self):
        r = "\n==============================================================================\n"
        r += ("obs_dim: %d\n" % self.obs_dim)
        r += ("obs_box: %s\n" % self.obs_box[0])
        r += ("         %s\n" % self.obs_box[1])
        r += ("act_dim: %d\n" % self.act_dim)
        r += ("act_box: %s\n" % self.act_box[0])
        r += ("         %s\n" % self.act_box[1])
        r += "==============================================================================\n\n"
        return r

