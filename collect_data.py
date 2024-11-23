from SteeringBehaviors import Wander
import SimulationEnvironment as sim

import numpy as np
import os
from filelock import FileLock

def collect_training_data(total_actions):
    # set-up environment
    sim_env = sim.SimulationEnvironment()

    # robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    path = 'saved/collect_data.npy'
    lock_path = 'saved/collect_data.npy.lock'

    while True:
        # network_params will be used to store your training data
        # a single sample will be composed of: sensor_readings, action, collision
        network_params = []

        # To ensure multithreaded data collection
        if os.path.exists(path):
            with FileLock(lock_path):
                load_data = np.load(path)
            network_params.extend(load_data.tolist())
            action_i = len(load_data)
            if action_i >= total_actions:
                break
        else:
            action_i = 0

        # steering_force is used for robot control only
        action, steering_force = steering_behavior.get_action(action_i, sim_env.robot.body.angle)

        sensor_readings = None
        collision = None
        for action_timestep in range(action_repeat):
            if action_timestep == 0:
                _, collision, sensor_readings = sim_env.step(steering_force)
            else:
                _, collision, _ = sim_env.step(steering_force)

            if collision:
                steering_behavior.reset_action()
                # this statement only EDITS collision of PREVIOUS action
                # if current action is very new.
                if action_timestep < action_repeat * .3 and action_i > 0: # in case prior action caused collision
                    network_params[-2][-1] = collision # share collision result with prior action
                break

        # Update network_params.
        network_params.append(sensor_readings.tolist() + [action, int(collision)])
        new_data = np.array(network_params)

        with FileLock(lock_path):
            np.save(path, new_data)

if __name__ == '__main__':
    collect_training_data(total_actions=50000)
