from gymnasium import register



def register_rover_environments():
    register(id="Rover4We-v2", entry_point='Rover.Environments.rover_4We_v2:RoverRobotrek4Wev2Env')
    # TODO: fix registering for rover2cam #register(id="Rover4We-v2", entry_point='Rover.Environments.rover_4We_')

