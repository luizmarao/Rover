from gymnasium import register
from Rover.Environments.rover_4We_v1 import RoverRobotrek4Wev1Env as roverv1



def register_rover_environments():
    register(id="Rover4We-v1", entry_point='Rover.Environments.rover_4We_v1:RoverRobotrek4Wev1Env')
    register(id="Rover4We-v2", entry_point='Rover.Environments.rover_4We_v1:RoverRobotrek4Wev2Env')

