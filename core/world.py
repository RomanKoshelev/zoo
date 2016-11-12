class World:
    def __init__(self):
        self.state = None

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *args):
        raise NotImplementedError

    def reset(self):
        self._reset()

    def _reset(self):
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
