from mj.agent import MujocoAgent


class ScorpionAgent(MujocoAgent):
    def provide_inputs(self, inputs_id):
        target = self.get_agent('target')
        if inputs_id == 'target_x':
            return target.sensor_val('x')
        if inputs_id == 'target_z':
            return target.sensor_val('z')
