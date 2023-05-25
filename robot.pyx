from pylab import *
from enum import IntEnum
from scipy.spatial.distance import euclidean

class SensorSide(IntEnum) :
    LEFT = 0
    RIGHT = 1

class Robot(object) :
    def __init__(self) :
        self.reset()
        self.lights = {}
        self.sensors = {}
        self.sensors_h = {}

    def reset(self) :
        self.RADIUS = 0.1
        self.MOTOR_SPEED = 1.0
        
        ## the robot's position in a 2D space
        self.x = 0.0
        self.y = 0.0
        ## orientation (which direction it is facing). 
        ## varies between 0 and 2*pi
        self.a = 0.0

        ## lists used to track robot's position history (used
        ## primarily for plotting)
        self.x_h = []
        self.y_h = []
        self.a_h = []

        # these are the velocity of the robot's two motors.  The
        # movement of the robot is then calculated as a function of
        # these two variables.
        self.lm = 0.0 # left motor
        self.rm = 0.0 # right motor

        ## motor histories
        self.lm_h = []
        self.rm_h = []

    def add_light(self,light) :
        if light.light_type not in self.lights.keys() :
            self.lights[light.light_type] = []
            self.sensors[light.light_type] = (0.0,0.0)
            self.sensors_h[light.light_type] = []
        
        self.lights[light.light_type].append(light)

    def update_sensors(self) :
        ## calculate sensor positions
        beta = np.pi / 4.0
        lsx = self.x + cos(self.a+beta)*self.RADIUS
        lsy = self.y + sin(self.a+beta)*self.RADIUS
        lsa = self.a + beta
        rsx = self.x + cos(self.a-beta)*self.RADIUS
        rsy = self.y + sin(self.a-beta)*self.RADIUS
        rsa = self.a - beta

        ## calculate light impacts on sensors
        for light_type in self.lights.keys() :
            ls = rs = 0.0
            for light in self.lights[light_type] :
                ls += light.impact_sensor(lsx,lsy,lsa)
                rs += light.impact_sensor(rsx,rsy,rsa)
            ls=min(1.0,ls)
            rs=min(1.0,rs)
            self.sensors[light_type] = (ls,rs)
            self.sensors_h[light_type].append((ls,rs))

    def is_close_to_any_light_or_the_robot(self,x,y,close):
        ## calculate light impacts on sensors
        if (self.x-x)**2 + (self.y-y)**2 < (4*close)**2 :
            return True
        
        for light_type in self.lights.keys() :
            for light in self.lights[light_type] :
                if (light.x-x)**2 + (light.y-y)**2 < close**2 :
                    return True
        return False
            
    def calculate_derivative(self) :
        ## Given the left and right motor values of the robot, this
        ## function calculates the rate at which the x and y position
        ## of the robot and its orientation are currently changing.
        self.update_sensors()
        self.lm_h.append(self.lm)
        self.rm_h.append(self.rm)
        
        self.dx = self.MOTOR_SPEED * cos(self.a)*(self.lm+self.rm)
        self.dy = self.MOTOR_SPEED * sin(self.a)*(self.lm+self.rm)
        self.da = self.MOTOR_SPEED * (self.rm-self.lm) / self.RADIUS
        
    def euler_update(self,DT=0.02) :
        ## these lists track the position and heading of the robot
        ## for plotting purposes
        self.x_h.append(self.x)
        self.y_h.append(self.y)
        self.a_h.append(self.a)

        #### INSERT EULER INTEGRATION HERE (START)
        self.x = self.x + self.dx * DT
        self.y = self.y + self.dy * DT
        self.a = self.a + self.da * DT
        #### INSERT EULER INTEGRATION HERE (END)

        WRAP = True
        if WRAP :
            ## periodic (wrap around) boundaries
            r = 1.5 # wrap_radius 

            if self.x > r :
                self.x -= 2*r
            if self.x < -r :
                self.x += 2*r
            if self.y > r :
                self.y -= 2*r
            if self.y < -r :
                self.y += 2*r

class Light(object) :
    def __init__(self,x,y,light_type) :
        """
        x,y -- position
            
        light_type -- a string or other unique identifier to allow
        different sensors to be sensitive to different 'types' of
        light. Could be 'FOOD' or 'RED', etc.

        """
        self.x = x
        self.y = y
        self.light_type = light_type

    def impact_sensor(self,sensor_x,sensor_y,sensor_angle) :
        accum = 0.0
        ## compensating for wrap around light viewing
        ## acting as if there are lights at all of these offsets
        # for lox,loy in [[-1.,0.],
        #                 [+1.,0.],
        #                 [0.,-1.],
        #                 [0.,+1.],
        #                 [0.,0.]] :
        for lox,loy in [[0.,0.]] : ## this just has one light (no wrap around lighting)
            lx = self.x + lox
            ly = self.y + loy
            
            dSq = (sensor_x - lx)**2 + (sensor_y - ly)**2

            # if the sensor were omnidirectional, its value would be
            falloff = 0.25 # lower number, sensors fall off faster
            omni = falloff/(falloff+dSq)

            # ## ... but instead, we are going to have a linear falloff
            # omni = max(0.0,1.0-dSq)
                        
            # calculate attenuation due to directionality
            # sensor to light unit vector
            s2l = [lx - sensor_x,
                   ly - sensor_y]
            s2l_mag = np.sqrt(s2l[0]**2 + s2l[1]**2)
            if s2l_mag > 0.0 :
                s2l = [v / s2l_mag for v in s2l]

            # sensor direction unit vector
            sd = [cos(sensor_angle),
                  sin(sensor_angle)]

            # positive set of dot product
            attenuation = max(0.0,s2l[0]*sd[0] + s2l[1]*sd[1])

            accum += omni * attenuation
        return accum


def test_directional_light_sensors() :
    r = Robot()
    lx=ly=0.5 # light position
    l = Light(0.5,0.5,'default')
    r.add_light(l)

    res = linspace(0,1,50)
    xs,ys = mesh = meshgrid(res,res)

    def f(coords) :
        r.x = coords[0]
        r.y = coords[1]
        r.update_sensors()
        return r.sensors['default'][SensorSide.LEFT]

    zs = apply_along_axis(f,0,mesh)
    print(shape(zs))
    imshow(zs,extent=[0,1,0,1],origin='lower')
    plot(lx,ly,'wo',label='light')
    xlabel('robot position')
    ylabel('robot position')
    title(f'Robot is facing to the right')
    legend()
    show()
    

if __name__ == '__main__' :
    #test_directional_light_sensors()

    for n_robots in range(10) : 
        duration = 50.0
        DT = 0.02
        iterations = int(np.round(duration/DT))

        robot = Robot()
        robot.x = np.random.randn()
        robot.y = np.random.randn()
        robot.a = np.random.rand()*np.pi*2.0
        light = Light(0,0,'default')
        robot.add_light(light)

        for iteration in range(iterations) :
            robot.calculate_derivative()
            robot.euler_update(DT=DT)

            ## these are the current state of the robot's sensors
            left_sensor = robot.sensors['default'][SensorSide.LEFT]
            right_sensor = robot.sensors['default'][SensorSide.RIGHT]
            
            #print(f'l:{left_sensor}\t r:{right_sensor}')

            ## NOT PARTICULARLY INTERESTING ROBOT
            robot.lm = 0.4 
            robot.rm = 0.5

            ## BRAITENBERG AGGRESSION

            ## BRAITENBERG LOVE


            

        plot(robot.x_h,
             robot.y_h,
             ',')

        plot(robot.x_h[-1],
             robot.y_h[-1],'ko',ms=3)

    plot(-999,-999,'k.',label=f'Robot Final Position')
    plot(0,0,',',label=f'Robot Trajectory')
    plot(0,0,'rx',label='Light Position')
    xlim(-3,3)
    ylim(-3,3)
    legend()
    gca().set_aspect('equal')
    show()

