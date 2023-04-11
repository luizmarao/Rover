function acker_experiment_struct = ackerman_avaliation(comment)
    

    if nargin ~= 1
        error('a comment must be entered')
    end
    if mj_connected == false
        mj_connect
    end

    mj_reset()

    state=mj_get_state();
    %[ghost-steer drive] 
    control = mj_get_control();
    
    s = -.30:.05:.30;
    steer = [];
    
    d = [-1.0000 -0.5000 -0.2000 0.2000 0.5000 1.0000];
    drive = repmat(d, 1, length(s));
    drive = cat(2, zeros(1, 1), drive);
    
    for i=1:length(s)
        steer = cat(2, steer, repmat(s(i),1,length(d)));
    end
    
    steer = cat(2, zeros(1, 1), steer);
    
    length(steer)
    length(drive)
    
    steer_index = -1
    last_steer_index = -1

    each_period = 300

    for i = 1:each_period*length(steer)
        state(end+1) = mj_get_state();

        steer_index = fix((i-1)/each_period)+1;
        
        mj_message(sprintf('steer = %6.3f drive = %6.3f', steer(steer_index), drive(steer_index)))
        
        control.ctrl = [steer(steer_index) drive(steer_index)];

        if steer_index ~= last_steer_index
            fprintf('i = %5d steer_index = %3d steer = %5.2f drive = %5.2f\n', i, steer_index, steer(steer_index), drive(steer_index));
            last_steer_index = steer_index;
        end

        mj_set_control(control)

        mj_step()
    end

    state(end+1) = mj_get_state();
        
    state = state(2:end)

    control.ctrl = [0 0];

    mj_set_control(control)

    mj_step()



    %% getting values

    model_xml = fileread('D:\Git\robottreking_evolution\gym\gym\envs\mujoco\assets\Rover4We-v1\rover-4-wheels-diff-acker.xml');

    [init, final]=regexp(model_xml,'solimp="(\d+\.\d+\s?)+');
    solimp = str2num(model_xml(init+8:final))

    [init, final]=regexp(model_xml,'solref="(\d+\.\d+\s?)+');
    solref = str2num(model_xml(init+8:final))


%     if exist('acker_experiment','var')
%         acker_experiment(end+1).about.solimp = solimp;
%         acker_experiment(end).about.solref = solref;
%     else
    acker_experiment_struct = [];
    acker_experiment_struct.about.tested_steer = steer;
    acker_experiment_struct.about.tested_drive = drive;
    acker_experiment_struct.about.each_period = each_period;
    
    acker_experiment_struct.about.solimp = solimp;
    acker_experiment_struct.about.solref = solref;
    
    acker_experiment_struct.about.xml_model = fileread('D:\Git\robottreking_evolution\gym\gym\envs\mujoco\assets\Rover4We-v1\rover-4-wheels-diff-acker.xml');
%     end

    i_f_l_steer_hinge   = 11;
    % for single front wheel
    i_f_r_steer_hinge   = 13;
    
    %%for double front wheel
    %i_f_r_steer_hinge   = 14;
    
    i_ghost_steer_hinge = 10;

    acker_experiment_struct.f_l_steer_hinge     = zeros(1,length(state));
    acker_experiment_struct.f_r_steer_hinge     = zeros(1,length(state));
    acker_experiment_struct.ghost_steer_hinge   = zeros(1,length(state));
    acker_experiment_struct.time                = zeros(1,length(state));

    for i = 1:length(state)
        acker_experiment_struct.f_l_steer_hinge(i)  = state(i).qpos(i_f_l_steer_hinge);
        acker_experiment_struct.f_r_steer_hinge(i)  = state(i).qpos(i_f_r_steer_hinge);
        acker_experiment_struct.ghost_steer_hinge(i)= state(i).qpos(i_ghost_steer_hinge);
        acker_experiment_struct.time(i)             = state(i).time;
    end

    pleft   = fliplr([ 9.3610018132620019e-005 2.0041497325730204e+000 1.4716131190668178e+000 -1.0929725380842041e+000 -3.2252012765547819e+000]);
    
    acker_experiment_struct.error_left = polyval(pleft , acker_experiment_struct.ghost_steer_hinge)-acker_experiment_struct.f_l_steer_hinge;
    acker_experiment_struct.about.max_error_left = max(abs(acker_experiment_struct.error_left));
    acker_experiment_struct.about.std_error_left =     std(acker_experiment_struct.error_left);
    
    pright  = fliplr([-1.7013070152971507e-004 2.0039379507086514e+000 -1.4684576386267671e+000 -1.0901354124362155e+000 3.1974185558159771e+000]);
    
    acker_experiment_struct.error_right = polyval(pright , acker_experiment_struct.ghost_steer_hinge)-acker_experiment_struct.f_r_steer_hinge;
    acker_experiment_struct.about.max_error_right = max(abs(acker_experiment_struct.error_right));
    acker_experiment_struct.about.std_error_right =     std(acker_experiment_struct.error_right);
    
    acker_experiment_struct.about.comment = comment
    











