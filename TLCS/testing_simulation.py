import traci
import numpy as np
import random
import timeit

# Phase codes based on baneswor_final.net.xml
# Your network has 4 phases in the tlLogic, we'll map actions to these phases
PHASE_0 = 0  # action 0 - phase 0 (duration 45s in default)
PHASE_1 = 1  # action 1 - phase 1 (duration 40s in default)
PHASE_2 = 2  # action 2 - phase 2 (duration 35s in default)
PHASE_3 = 3  # action 3 - phase 3 (duration 95s in default)


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1  # dummy init

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        Adapted for Baneswor network: DR2, RU1, UL2, LD1
        """
        incoming_roads = ["DR2", "RU1", "UL2", "LD1"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:  # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        Note: Your Baneswor network has yellow phases built into the tlLogic.
        For simplicity, we can either skip yellow or use a brief transition.
        Here we'll use a simple approach - just switch directly.
        """
        # Option: Could implement yellow transition if needed
        # For now, the yellow_duration will just be a brief pause
        pass


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        Maps actions to the 4 phases defined in baneswor_final.net.xml
        """
        if action_number == 0:
            traci.trafficlight.setPhase("J1", PHASE_0)
        elif action_number == 1:
            traci.trafficlight.setPhase("J1", PHASE_1)
        elif action_number == 2:
            traci.trafficlight.setPhase("J1", PHASE_2)
        elif action_number == 3:
            traci.trafficlight.setPhase("J1", PHASE_3)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        Adapted for Baneswor network: DR2, RU1, UL2, LD1
        """
        halt_DR2 = traci.edge.getLastStepHaltingNumber("DR2")
        halt_RU1 = traci.edge.getLastStepHaltingNumber("RU1")
        halt_UL2 = traci.edge.getLastStepHaltingNumber("UL2")
        halt_LD1 = traci.edge.getLastStepHaltingNumber("LD1")
        queue_length = halt_DR2 + halt_RU1 + halt_UL2 + halt_LD1
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        Adapted for Baneswor network with lanes: DR2, RU1, UL2, LD1
        
        Lane structure from baneswor_final.net.xml:
        - DR2: 3 lanes (DR2_0, DR2_1, DR2_2)
        - RU1: 5 lanes (RU1_0, RU1_1, RU1_2, RU1_3, RU1_4)
        - UL2: 3 lanes (UL2_0, UL2_1, UL2_2)
        - LD1: 5 lanes (LD1_0, LD1_1, LD1_2, LD1_3, LD1_4)
        
        We'll map these to 8 lane groups to maintain 80 states (8 groups × 10 cells)
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            
            # Your network edges are ~180-186m long
            # We'll use 200m as max for consistency with original code
            lane_pos = 200 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located
            # Mapping Baneswor lanes to 8 lane groups (0-7) to get 80 total states
            if lane_id == "DR2_0" or lane_id == "DR2_1":
                lane_group = 0
            elif lane_id == "DR2_2":
                lane_group = 1
            elif lane_id == "RU1_0" or lane_id == "RU1_1" or lane_id == "RU1_2":
                lane_group = 2
            elif lane_id == "RU1_3" or lane_id == "RU1_4":
                lane_group = 3
            elif lane_id == "UL2_0" or lane_id == "UL2_1":
                lane_group = 4
            elif lane_id == "UL2_2":
                lane_group = 5
            elif lane_id == "LD1_0" or lane_id == "LD1_1" or lane_id == "LD1_2":
                lane_group = 6
            elif lane_id == "LD1_3" or lane_id == "LD1_4":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two position IDs to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



