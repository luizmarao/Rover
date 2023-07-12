from gymnasium import register



def register_rover_environments():
    register(id="Rover4We-v2", entry_point='Rover.Environments.rover_4We_v2:RoverRobotrek4Wev2Env')
    register(id="Rover4WeRedObs-v0", entry_point='Rover.Environments.rover_4We_v2:RoverRobotrek4Wev2RedObsEnv')
    register(id="Rover4WeEncodedVision-v0", entry_point='Rover.Environments.rover_4We_encoded_vision'
                                                        ':Rover4WeEncodedVisionv0Env')
    register(id="Rover4WePCAVision-v0", entry_point='Rover.Environments.rover_4We_pca_vision'
                                                        ':Rover4WePCAVisionv0Env')
    register(id="Rover4WeDoubleCameraPacked-v0", entry_point='Rover.Environments.rover_4We_DoubleCam'
                                                        ':Rover4WeDoubleCameraPackedv0Env')
    register(id="Rover4WeDoubleCameraFused-v0", entry_point='Rover.Environments.rover_4We_DoubleCam'
                                                             ':Rover4WeDoubleCameraFusedv0Env')
    register(id="Rover4W-v1", entry_point='Rover.Environments.rover_4W:Rover4Wv1Env')
    register(id="Rover4WRedObs-v0", entry_point='Rover.Environments.rover_4W:Rover4Wv1RedObsEnv')
    register(id="Rover4WMetaLearning-v0", entry_point='Rover.Environments.rover_4W_MetaLearning:RoverMetaLearningEnv')
