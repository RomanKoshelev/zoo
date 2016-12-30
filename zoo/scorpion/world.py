from mj.world import MujocoWorld


class ScorpionWorld(MujocoWorld):
    def provide_inputs(self, inputs_id):
        ball = self.get_agent('ball')
        if ball is not None:
            if inputs_id == 'ball_x':
                return ball.get_observation(ball.sensor_name('x'))['get_val']()
            if inputs_id == 'ball_y':
                return ball.get_observation(ball.sensor_name('y'))['get_val']()
            if inputs_id == 'ball_z':
                return ball.get_observation(ball.sensor_name('z'))['get_val']()
        return [None]
