from mj.world import MujocoWorld


class ScorpionWorld(MujocoWorld):
    def provide_inputs(self, inputs_id):
        ball = self.get_agent('ball')
        if inputs_id == 'ball_x':
            return ball.sensor_val('x')
        if inputs_id == 'ball_y':
            return ball.sensor_val('y')
        if inputs_id == 'ball_z':
            return ball.sensor_val('z')
