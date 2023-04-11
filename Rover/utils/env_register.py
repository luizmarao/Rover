from gymnasium import register
import Environments.rover_4We_v1 as roverenvs



def register_rover_environments():
    register(id="Rover4We-v1", entry_point='Environments.rover_4We_v1.roverenvs.RoverRobotrek4Wev1Env')

