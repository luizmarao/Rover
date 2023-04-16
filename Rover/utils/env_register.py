from gymnasium import register



def register_rover_environments():
    register(id="Rover4We-v1", entry_point='Rover.Environments.rover_4We_v1:RoverRobotrek4Wev1Env')
    # TODO: fix registering for rover2cam #register(id="Rover4We-v2", entry_point='Rover.Environments.rover_4We_')

