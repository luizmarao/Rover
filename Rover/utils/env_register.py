from gymnasium import register



def register_rover_environments():
    register(id="Rover4We-v2", entry_point='Rover.Environments.rover_4We_v2:RoverRobotrek4Wev2Env')
    register(id="Rover4WeEncodedVision-v0", entry_point='Rover.Environments.rover_4We_encoded_vision'
                                                        ':Rover4WeEncodedVisionv0Env')
    # TODO: fix registering for rover2cam #register(id="Rover4We-v2", entry_point='Rover.Environments.rover_4We_')

