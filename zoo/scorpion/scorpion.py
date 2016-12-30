from mj.agent import MujocoAgent


class ScorpionAgent(MujocoAgent):
    def provide_inputs(self, inputs_id):
        target = self.get_agent('target')
        if target is not None:
            if inputs_id == 'target_x':
                return target.get_observation(target.sensor_name('x'))['get_val']()
            if inputs_id == 'target_z':
                return target.get_observation(target.sensor_name('z'))['get_val']()
        return [None]
