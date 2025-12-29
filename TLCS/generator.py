import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        Adapted for Baneswor network with edges: DR2, RU1, UL2, LD1 (incoming)
        and DL2, LU1, UR2, RD1 (outgoing)
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

    <!-- Routes for Baneswor Network -->
    <!-- From DR2 (Down-Right, coming from South) -->
    <route id="DR2_LU1" edges="DR2 LU1"/>
    <route id="DR2_UR2" edges="DR2 UR2"/>
    <route id="DR2_RD1" edges="DR2 RD1"/>
    
    <!-- From RU1 (Right-Up, coming from East) -->
    <route id="RU1_DL2" edges="RU1 DL2"/>
    <route id="RU1_LU1" edges="RU1 LU1"/>
    <route id="RU1_UR2" edges="RU1 UR2"/>
    
    <!-- From UL2 (Up-Left, coming from North) -->
    <route id="UL2_RD1" edges="UL2 RD1"/>
    <route id="UL2_DL2" edges="UL2 DL2"/>
    <route id="UL2_LU1" edges="UL2 LU1"/>
    
    <!-- From LD1 (Left-Down, coming from West) -->
    <route id="LD1_UR2" edges="LD1 UR2"/>
    <route id="LD1_RD1" edges="LD1 RD1"/>
    <route id="LD1_DL2" edges="LD1 DL2"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        # South to North (DR2 to UR2)
                        print('    <vehicle id="DR2_UR2_%i" type="standard_car" route="DR2_UR2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        # East to West (RU1 to LU1)
                        print('    <vehicle id="RU1_LU1_%i" type="standard_car" route="RU1_LU1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        # North to South (UL2 to DL2)
                        print('    <vehicle id="UL2_DL2_%i" type="standard_car" route="UL2_DL2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        # West to East (LD1 to RD1)
                        print('    <vehicle id="LD1_RD1_%i" type="standard_car" route="LD1_RD1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:  # car that turn - 25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source & destination
                    if route_turn == 1:
                        # South to West (DR2 to LU1) - Left turn
                        print('    <vehicle id="DR2_LU1_%i" type="standard_car" route="DR2_LU1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        # South to East (DR2 to RD1) - Right turn
                        print('    <vehicle id="DR2_RD1_%i" type="standard_car" route="DR2_RD1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        # East to South (RU1 to DL2) - Left turn
                        print('    <vehicle id="RU1_DL2_%i" type="standard_car" route="RU1_DL2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        # East to North (RU1 to UR2) - Right turn
                        print('    <vehicle id="RU1_UR2_%i" type="standard_car" route="RU1_UR2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        # North to East (UL2 to RD1) - Left turn
                        print('    <vehicle id="UL2_RD1_%i" type="standard_car" route="UL2_RD1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        # North to West (UL2 to LU1) - Right turn
                        print('    <vehicle id="UL2_LU1_%i" type="standard_car" route="UL2_LU1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        # West to North (LD1 to UR2) - Left turn
                        print('    <vehicle id="LD1_UR2_%i" type="standard_car" route="LD1_UR2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        # West to South (LD1 to DL2) - Right turn
                        print('    <vehicle id="LD1_DL2_%i" type="standard_car" route="LD1_DL2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)