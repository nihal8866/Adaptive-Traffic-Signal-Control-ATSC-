from __future__ import absolute_import
from __future__ import print_function

import os
import traci
import numpy as np
import timeit
from shutil import copyfile

from generator import TrafficGenerator
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


class FixedTimeSimulation:
    def __init__(self, TrafficGen, sumo_cmd, max_steps):
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._reward_episode = []
        self._queue_length_episode = []
        self._step = 0
        
    def run(self, episode):
        """
        Run simulation with SUMO's built-in fixed-time control
        """
        start_time = timeit.default_timer()
        
        # Generate the same traffic pattern
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating with Fixed-Time Control...")
        
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        
        # Let SUMO run with its default fixed-time control
        while self._step < self._max_steps:
            traci.simulationStep()
            self._step += 1
            
            # Collect metrics
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait
            self._reward_episode.append(reward)
            
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)
            
            old_total_wait = current_total_wait
        
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        
        return simulation_time
    
    def _collect_waiting_times(self):
        incoming_roads = ["DR2", "RU1", "UL2", "LD1"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time
    
    def _get_queue_length(self):
        halt_DR2 = traci.edge.getLastStepHaltingNumber("DR2")
        halt_RU1 = traci.edge.getLastStepHaltingNumber("RU1")
        halt_UL2 = traci.edge.getLastStepHaltingNumber("UL2")
        halt_LD1 = traci.edge.getLastStepHaltingNumber("LD1")
        queue_length = halt_DR2 + halt_RU1 + halt_UL2 + halt_LD1
        return queue_length
    
    @property
    def queue_length_episode(self):
        return self._queue_length_episode
    
    @property
    def reward_episode(self):
        return self._reward_episode


if __name__ == "__main__":
    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    
    # Create output directory
    plot_path = os.path.join(os.getcwd(), 'comparison', 'fixed_time_baseline_2000', '')
    os.makedirs(plot_path, exist_ok=True)
    
    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )
    
    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
    
    Simulation = FixedTimeSimulation(
        TrafficGen,
        sumo_cmd,
        config['max_steps']
    )
    
    print('\n----- Fixed-Time Baseline Test')
    simulation_time = Simulation.run(config['episode_seed'])
    print('Simulation time:', simulation_time, 's')
    
    print("----- Fixed-time results saved at:", plot_path)
    
    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))
    
    Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue length (vehicles)')
    
    # Calculate statistics
    avg_queue = np.mean(Simulation.queue_length_episode)
    max_queue = np.max(Simulation.queue_length_episode)
    total_reward = np.sum(Simulation.reward_episode)
    
    print(f"\n----- Statistics:")
    print(f"Average Queue Length: {avg_queue:.2f} vehicles")
    print(f"Maximum Queue Length: {max_queue} vehicles")
    print(f"Total Cumulative Reward: {total_reward:.0f}")