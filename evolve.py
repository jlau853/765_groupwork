from pylab import *
import os,sys
from scipy.spatial.distance import euclidean

import pyximport; pyximport.install(language_level=3)
from robot import Robot,Light
from seth_controller import SethController, EntityTypes, ENTITY_RADIUS
from plotting import plot_state_history,fitness_plots,plot_population_genepool

from multiprocessing import Pool
plt.switch_backend('agg')

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

## where to write simulation output to
savepath = os.path.abspath('./output/')

TEST_GA = False

DRAW_EVERY_NTH_GENERATION = 5

## EVOLUTION PARAMETERS
N_TRIALS = 5
POP_SIZE = 25
generation_index = 0
SITUATION_DURATION = 15.0
DT = 0.02
N_STEPS = int(SITUATION_DURATION / DT) ## the maximum number of steps per trial

if TEST_GA :
    N_STEPS = 1
    N_TRIALS = 1

## THE POPULATION
pop = [SethController() for _ in range(POP_SIZE)] ## the evolving population (a list of SethControllers)

## This keeps track of the fitness of the entire population since the start of the evolution.
## It is plotted in fitness_history.png
pop_fit_history = []

def random_light_position(robot) :    
    x = np.random.rand()*2.0-1.0
    y = np.random.rand()*2.0-1.0
    # make sure new position is not too close to other lights or the robot
    while robot.is_close_to_any_light_or_the_robot(x,y,2.0 * ENTITY_RADIUS) :
        x = np.random.rand()*2.0-1.0
        y = np.random.rand()*2.0-1.0
    return x,y

def simulate_trial(controller,trial_index,generating_animation=False) :
    """
    controller       -- the controller we are simulating

    trial_index -- we evaluate fitness by taking the average of N
                   trials. This argument tells us which trial we are
                   currently simulating

    """
    # ## reset the seed to make randomness of environment consistent for this generation
    np.random.seed(generation_index*10+trial_index)

    #### initialize the simulation    
    current_time = 0.0    
    score = 0.0    

    ## reset the robot
    robot = Robot()
    robot.x = 0.0
    robot.y = 0.0
    robot.a = 0.0

    ## get the controller for the robot from the population
    controller.trial_data = {}

    food_entities = []
    water_entities = []
    trap_entities = []

    ## reset the environment
    for entity_type in EntityTypes:
        n = {
            EntityTypes.FOOD  : 2,  ## how many of each entity type to create
            EntityTypes.WATER : 2,
            EntityTypes.TRAP  : 2,
        }
        for _ in range(n[entity_type]) :
            x,y = random_light_position(robot)
            entity_light = Light(x,y,entity_type)
            robot.add_light(entity_light)

            if entity_type == EntityTypes.FOOD :
                food_entities.append(entity_light)
            if entity_type == EntityTypes.WATER :
                water_entities.append(entity_light)
            if entity_type == EntityTypes.TRAP :
                trap_entities.append(entity_light)
    
    ## batteries
    water_b = 1.0
    food_b = 1.0
    
    ## these variables keep track of things we want to plot later
    controller.trial_data['sample_times'] = [] 
    controller.trial_data['water_battery_h'] = []
    controller.trial_data['food_battery_h'] = []
    controller.trial_data['score_h'] = []
    controller.trial_data['eaten_FOOD_positions']  = [] 
    controller.trial_data['eaten_WATER_positions'] = []
    controller.trial_data['eaten_TRAP_positions']  = []
    controller.trial_data['FOOD_positions'] = []
    controller.trial_data['WATER_positions'] = []
    controller.trial_data['TRAP_positions'] = []
    
    for iteration in range(N_STEPS) :
        ## keep track of battery states for plotting later
        controller.trial_data['sample_times'].append(current_time)
        controller.trial_data['water_battery_h'].append(water_b)
        controller.trial_data['food_battery_h'].append(food_b)
        controller.trial_data['score_h'].append(score)

        if generating_animation :
            ##used in animation
            controller.trial_data[f'FOOD_positions'].append( [(l.x,l.y) for l in food_entities] )
            controller.trial_data[f'WATER_positions'].append( [(l.x,l.y) for l in water_entities] )
            controller.trial_data[f'TRAP_positions'].append( [(l.x,l.y) for l in trap_entities] )
        
        ## each iteration, the time moves forward by the time-step, 'DT'
        current_time += DT
        
        ## the battery states steadily drain at a constant rate
        DRAIN_RATE = 0.2
        water_b = water_b - DT*DRAIN_RATE
        food_b  = food_b - DT*DRAIN_RATE

        # water_b -= (robot.lm**2) * DT * 0.01
        # food_b  -= (robot.rm**2) * DT * 0.01
        
        score += (water_b * food_b) * DT
        
        #### interaction between body and controller
        ## tell the controller what the states of its various sensors are
        for entity_type in EntityTypes:
            controller.set_sensor_states(entity_type,robot.sensors[entity_type])
        ## tell the robot body what the controller says its motors should be
        robot.lm,robot.rm = controller.get_motor_output((food_b,water_b))
        
        robot.calculate_derivative() ## calculate the way that the robot's state it changing
        robot.euler_update(DT=DT)    ## apply that state change (Euler integration step)

        #### check if robot has collided with FOOD, WATER or TRAP and
        #### update batteries, etc.  accordingly
        ## check for FOOD collisions
        for light in robot.lights[EntityTypes.FOOD] :
            if (robot.x - light.x)**2 + (robot.y - light.y)**2 < ENTITY_RADIUS**2 :
                food_b += 20.0*DT
                controller.trial_data['eaten_FOOD_positions'].append( (light.x,light.y) )
                light.x,light.y = random_light_position(robot) ## relocate entity

        ## check for WATER collisions
        for light in robot.lights[EntityTypes.WATER] :
            if (robot.x - light.x)**2 + (robot.y - light.y)**2 < ENTITY_RADIUS**2 :
                water_b += 20.0*DT
                controller.trial_data['eaten_WATER_positions'].append( (light.x,light.y) )
                light.x,light.y = random_light_position(robot) ## relocate entity

        ## check for TRAP collisions                
        for light in robot.lights[EntityTypes.TRAP] :
            if (robot.x - light.x)**2 + (robot.y - light.y)**2 < ENTITY_RADIUS**2 :
                food_b -= 50.0*DT
                water_b -= 50.0*DT
                score = 0.0
                controller.trial_data['eaten_TRAP_positions'].append( (light.x,light.y) )
                light.x,light.y = random_light_position(robot) ## relocate entity

        ## DEATH -- if either of the batteries reaches 0, the trial is over
        if food_b < 0.0 or water_b < 0.0 :
            food_b = water_b = 0.0
            break            

    ## record the position of the entities still not eaten at end of trial (used for plotting)
    controller.trial_data['uneaten_FOOD_positions']  = [(l.x,l.y) for l in robot.lights[EntityTypes.FOOD]]
    controller.trial_data['uneaten_WATER_positions'] = [(l.x,l.y) for l in robot.lights[EntityTypes.WATER]]
    controller.trial_data['uneaten_TRAP_positions']  = [(l.x,l.y) for l in robot.lights[EntityTypes.TRAP]]
    controller.trial_data['robot'] = robot

    if TEST_GA :
        score = np.mean(controller.genome)
    
    return score

def evaluate_fitness(controller) :
    """An evaluation of an individual's fitness is the average of its
    performance in N_TRIAL simulations.

    ind -- the controller that is being evaluated

    """
    
    trial_scores = [simulate_trial(controller,trial_index) for trial_index in range(N_TRIALS)]
    controller.fitness = np.mean(trial_scores)
    
    return controller

def generation() :
    global pop,generation_index
    
    ## parallel evaluation of fitnesses (in parallel using multiprocessing)
    with Pool() as p:
        pop = p.map(evaluate_fitness, pop)

    # ## sequential evaluation of fitness
    # pop = [evaluate_fitness(controller) for controller in pop]

    ## the fitness of every individual controller in the population
    fitnesses = [r.fitness for r in pop]
    ## we track the fitness of every individual at every generation for plotting
    pop_fit_history.append(sorted(np.array(fitnesses)))


    # ## every nth generation, we plot the trajectories of the best and
    # ## worst individual
    if (generation_index % DRAW_EVERY_NTH_GENERATION) == 0 :
        best_index = np.argmax(fitnesses)
        plot_state_history(savepath,pop[best_index],'best')
        pop[best_index].plot_links('best')

        np.save(os.path.join(savepath,'best_genome.npy'),pop[best_index].genome)

        worst_index = np.argmin(fitnesses)
        plot_state_history(savepath,pop[worst_index],'worst')
        pop[worst_index].plot_links('worst')

        # ## can plot others too, but actually takes a fair amount of
        # ## time, so leave commented out unless curious
        # for ind_i in np.arange(POP_SIZE)[::5] :
        #     plot_state_history(os.path.join(savepath,'everyone/'),pop[ind_i],f'everyone/i{ind_i}')
        #     pop[worst_index].plot_links(f'everyone/i{ind_i}')

    ######################################################################
    #### USE FITNESS PROPORTIONATE SELECTION TO CREATE THE NEXT GENERATION
    def weighted_choice(weights):
        totals = []
        running_total = 0

        for w in weights:
            running_total += w
            totals.append(running_total)

        rnd = np.random.rand() * running_total
        for i, total in enumerate(totals):
            if rnd < total:
                return i

    #### Calcuate ps, the probabiltiy that each individual has for
    #### being a parent
    
    ## normalize the distribution of fitnesses to lie between 0 and 1
    f_normalized = np.array([x for x in fitnesses])
    f_normalized -= min(f_normalized)
    if max(f_normalized) > 0.0 :
        f_normalized /= max(f_normalized)
    sum_f = max(0.01,sum(f_normalized))

    ## ps: the probability of being selected as a parent of each
    ## individual in the next generation
    ps = [f/sum_f for f in f_normalized]
    
    #### NOW ACTUALLY CREATE THE NEXT GENERATION

    ## "elitism" : we seed the next generation of a copy of the best
    ## individual from the previous generation
    best_index = np.argmax(fitnesses)
    best_individual = pop[best_index]
    next_generation = [ SethController(genome = best_individual.genome) ]

    ## ...and then populate the rest of the next generation by selecting
    ## parents in a weighted manner such that higher fitness parents have
    ## a higher chance of being selected.
    while len(next_generation) < POP_SIZE :
        a_i = weighted_choice(ps)
        b_i = weighted_choice(ps)
        mama = pop[a_i]
        dada = pop[b_i]
        baby = mama.procreate_with(dada)
        next_generation.append(baby)

    ## replace the old generation with the new
    pop = next_generation

    print(f'GENERATION # {generation_index}.\t(mean/min/max)'+
          f'({np.mean(fitnesses):.4f}/{np.min(fitnesses):.4f}/{np.max(fitnesses):.4f})')
    generation_index += 1

    
def evolve() :
    global fitnesses,generation_index

    print(f'Every generation is {POP_SIZE*N_TRIALS} fitness evaluations.')
    
    while True :
        fitnesses = generation()

        if generation_index % DRAW_EVERY_NTH_GENERATION == 0 :
            fitness_plots(savepath,pop_fit_history)
            plot_population_genepool(savepath,pop)


if __name__ == '__main__' :
    evolve()
