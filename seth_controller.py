from pylab import *
import copy

from enum import Enum,IntEnum

ENTITY_RADIUS = 0.1

class EntityTypes(IntEnum):
    FOOD  = 0
    WATER = 1
    TRAP  = 2

class Sides(IntEnum):
    LEFT_SIDE   = 0 # all connections are ipsilateral
    RIGHT_SIDE = 1  # 
    
class SMLink(object):
    """ A single sensor to motor link. A SethController involves several of these."""
    
    N_MID_POINTS = 2
    N_GENES = 2 + N_MID_POINTS*2 + 3  # 2 x-values (the mid points); 4 y-values; 3 battery infl. parameters

    def __init__(self):
        pass

    def set_genome(self,genome) :
        """Takes a genome of length N_GENES, with every element inbetween 0 and 1. 
        Uses those values to create a SMLink. 

        The first N_MID_POINTS values specify the x-positions of the piecewise control points.
        The next 2+N_MID_POINTS values specify the y-positions of the piecewise control points.
        The last three values specify 

          -- which battery the link responds to
          -- parameter of the linear (additive) battery response
          -- parameter of the scalar (muliplicative) battery response

        """
        NXG = SMLink.N_MID_POINTS
        NYG = SMLink.N_MID_POINTS+2
        self.xs = np.array([0.0] + sorted(genome[:NXG]) + [1.0])
        self.ys = np.array(genome[NXG:NXG+NYG]) *2.0 - 1.0
        self.battery_sensitivity = int(genome[NXG+NYG] < 0.5) # 0 is food battery, 1 is water
        self.O = 2.0*genome[NXG+NYG+1] - 1.0 ## \in (-1,1)        
        self.S = genome[NXG+NYG+2]           ## \in (0,1)

        ## used for plotting which battery this link is sensitive to
        self.bsense = ['F','W'][self.battery_sensitivity] 

    def output(self, sensory_state, battery_states) :
        # the unscaled value shows what this link would output if it
        # were not affected by battery level. It is not used during an
        # actual simulation, but is useful when visualizing the links
        
        out = np.interp(sensory_state,self.xs,self.ys)
        unscaled = out

        ## apply effect of battery as described in (Seth,2004)
        b = battery_states[self.battery_sensitivity]
        b = max(min(b,1.0),0.0)
        out = out + b * self.O * 2.0
        out = out + out * (b*2.0 - 1.0) * self.S

        ## what does this do?
        out=max(min(out,1.0),-1.0)
        
        return out,unscaled

class SethController(object):
    def __init__(self, genome=None, n_sensitivities=3, n_motors=2) :
        self.n_sensitivities = n_sensitivities
        self.n_motors = n_motors
        self.SYMMETRIC = True
        if self.SYMMETRIC :
            n_sides = 1
        else :
            n_sides = 2

        self.n_links = n_sensitivities * n_sides ## all links are ipsilateral
        
        self.N_GENES = self.n_links * SMLink.N_GENES
        
        if genome is None :
            ## create a new random genome (used to seed the initial population)
            self.genome = np.random.rand(self.N_GENES)
        else :
            ## otherwise set the controller's genome to the argument
            self.genome = genome
            
        ## self.links is an array of links indexed by
        ## [Side,EntityType] so you can get the link that
        ## specifies the left link for the FOOD sensitivity
        ## by writing `self.links[Sides.LEFT_SIDE,Thing.FOOD]`
        self.links = np.array([
            [SMLink() for s in range(3)]
            for ipsicontra in range(n_sides)])
            
        self.genome_to_links()
        self.sensor_states = {}

    def genome_to_links(self) :
        """Translates the genome into a collection of SMLinks that can be used
        to translate sensory state into motor output.

        """
        links = self.links.view().reshape(self.n_links)

        for link_i in range(self.n_links):
            link_genotype = self.genome[link_i*SMLink.N_GENES:(link_i+1)*SMLink.N_GENES]
            links[link_i].set_genome(link_genotype)

    def set_sensor_states(self,entity_type,sensor_values):
        """Used to couple the robot's sensors to this controller.

        -- entity_type: the category of entity beind sensed (e.g. EntityTypes.FOOD)

        -- sensor_values: a tuple indicating the excitation of the (left,right) sensor of that type

        a call of set_sensor_states(EntityTypes.WATER,[0,0.9]) is a
        way of telling the controller that currently the left WATER
        sensor is minimally excited (has a value of 0), while the
        right one is highly excited (has a value of 0.9).

        """
        self.sensor_states[entity_type] = sensor_values

    def get_motor_output(self,battery_states) :
        """Calculates the current motor output given the current sensor state
        (as last set by calls to set_sensor_states) and given the
        battery states passed in the argument.

        """
        lm = rm = 0.0

        LEFT = 0
        RIGHT = 1
        for entity_type in EntityTypes:
            lm += self.links[Sides.LEFT_SIDE,entity_type].output(
                self.sensor_states[entity_type][LEFT],
                battery_states)[0]

            if self.SYMMETRIC :
                ## use the same link as the left side but with the
                ## right sensor for the right motor
                rm += self.links[Sides.LEFT_SIDE,entity_type].output(
                    self.sensor_states[entity_type][RIGHT],
                    battery_states)[0]

            else :
                rm += self.links[Sides.RIGHT_SIDE,entity_type].output(
                    self.sensor_states[entity_type][RIGHT],
                    battery_states)[0]
            
        return lm,rm
        
    def procreate_with(self,other_controller,
                       mu=0.01,mu2=0.01) :
        mama = self
        dada = other_controller
                
        ## recombine genes (take half from mama, half from dada)
        baby_genome = np.where(np.random.rand(len(mama.genome))<0.5,
                               mama.genome,dada.genome)

        ## mutate genes
        baby_genome += mu*np.random.randn(len(mama.genome))
        def bounce(y) :
            if y > 1.0 :
                y = 1.0-(y-1.0)
            elif y < 0.0:
                y = -y
            return y

        baby_genome = [bounce(x) for x in baby_genome]

        if np.random.rand() < mu2 :
            gene_index = np.random.randint(len(mama.genome))
            baby_genome[gene_index] = np.random.rand()        

        baby = SethController(genome=baby_genome)
        return baby


    def plot_links(self,file_prefix) :
        figure()
        if self.SYMMETRIC :
            n_sides = 1
            sides = [Sides.LEFT_SIDE]
        else :
            n_sides = 2
            sides = [Sides.LEFT_SIDE,Sides.RIGHT_SIDE]
        
        for side in sides :
            for sensitivity in EntityTypes :
                l = self.links[side,sensitivity]
                subplot2grid((3,n_sides),(sensitivity,side))
                xs = linspace(0,1,101)
                highbat_outs = []
                lowbat_outs = []
                unscaled_outs = []
                for x in xs :
                    highbat_out,unscaled_out = l.output(x,[1.0,1.0])
                    lowbat_out,unscaled_out = l.output(x,[0.0,0.0])
                    highbat_outs.append(highbat_out)
                    lowbat_outs.append(lowbat_out)
                    unscaled_outs.append(unscaled_out)
                plot(xs,highbat_outs,label='high battery')
                plot(xs,lowbat_outs,label='low battery')
                plot(xs,unscaled_outs,label='high battery')
                text(1.02,-0.98,l.bsense[0],color='r')
                xlim(0,1)
                ylim(-1.05,1.05)
                ylabel(sensitivity.name)
                if sensitivity == 0:
                    if not self.SYMMETRIC :
                        title(side.name)
                    else :
                        title('Symmetric Ipsilateral Links')

        tight_layout()
        savefig(f'output/{file_prefix}_links.png',dpi=120)
        close()
        

if __name__ == '__main__' :
    l = SMLink()
    genome = np.random.rand(SMLink.N_GENES) ## a random genome
    #    genome = [ ] # UNCOMMENT AND EDIT THIS LINE TO SPECIFY YOUR GENOME
    l.set_genome(genome)
    xs = linspace(0,1,101)

    highbat_outs = []
    lowbat_outs = []
    unscaled_outs = []
    for x in xs :
        highbat_out,unscaled_out = l.output(x,[1.0,1.0])
        lowbat_out,unscaled_out = l.output(x,[0.0,0.0])
        highbat_outs.append(highbat_out)
        lowbat_outs.append(lowbat_out)
        unscaled_outs.append(unscaled_out)
    plot(xs,highbat_outs,label='high battery')
    plot(xs,lowbat_outs,label='low battery')
    plot(xs,unscaled_outs,label='no battery influence')
    xlabel('ipsilateral sensor')
    ylabel('ipsilateral motor output')
    xlim(0,1)
    ylim(-1.05,1.05)
    legend()
    show()
