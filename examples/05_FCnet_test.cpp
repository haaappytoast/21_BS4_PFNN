#include <aPfnn.h>
#include <iostream>

#define in_size 234
#define out_size 279
class MyApp : public agl::App
{
public:
    
    void draw_velocity(Vec3 position, Vec3 velocity, float scale)
    {
        Vec3 start_pos = position;
        Vec3 end_pos = position + velocity;

        Vec3 x_axis = (end_pos - start_pos).normalized();
        // Vec3 x_inv = -x_axis;
        // Vec3 y_axis = x_axis.cross(x_inv);
        // std::cout << "CROSS: " << y_axis << std::endl;
        // y_axis = y_axis.normalized();
        Vec3 y_axis = Vec3::UnitY();
        Vec3 z_axis = x_axis.cross(y_axis);

        Mat4 trf = Mat4::Identity();
        trf.col(0).head<3>() = x_axis;
        trf.col(1).head<3>() = y_axis;
        trf.col(2).head<3>() = z_axis;
        trf.col(3).head<3>() = position;
        //std::cout << trf << std::endl;

        float offset = scale * 0.08f * velocity.norm();
        Mat4 preTrf = Mat4::Identity();
        preTrf.col(3).head<3>().x() = -(offset / 2.0f);

        Mat4 final_trf = trf * preTrf;

        agl::Render::cube()
            ->transform(final_trf)
            ->scale(offset, 0.02f, 0.02f)
            ->color(0, 1, 0)
            ->debug(true)
            ->draw();
    }
    
    void draw_direction(Vec3 position, Vec3 velocity, float scale)
    {
        Vec3 start_pos = position;
        Vec3 end_pos = position + velocity;

        Vec3 x_axis = (end_pos - start_pos).normalized();
        // Vec3 x_inv = -x_axis;
        // Vec3 y_axis = x_axis.cross(x_inv);
        // std::cout << "CROSS: " << y_axis << std::endl;
        // y_axis = y_axis.normalized();
        Vec3 y_axis = Vec3::UnitY();
        Vec3 z_axis = x_axis.cross(y_axis);

        Mat4 trf = Mat4::Identity();
        trf.col(0).head<3>() = x_axis;
        trf.col(1).head<3>() = y_axis;
        trf.col(2).head<3>() = z_axis;
        trf.col(3).head<3>() = position;
        //std::cout << trf << std::endl;

        float offset = scale * 0.08f * velocity.norm();
        Mat4 preTrf = Mat4::Identity();
        preTrf.col(3).head<3>().x() = (offset / 2.0f);

        Mat4 final_trf = trf * preTrf;

        agl::Render::cube()
            ->transform(final_trf)
            ->scale(offset, 0.01f, 0.01f)
            ->color(1, 0, 0)
            ->debug(true)
            ->draw();
    }
    
    void draw_local_coord(Mat4 trf, float scale = 1.0f)
    {
        //color of x-axis; R
        //color of y-axis; G
        //color of z-axis; B
        float offset = 0.08f;

        Mat4 xOffset = Mat4::Identity();
        xOffset(0, 3) = scale * offset;

        Mat4 yOffset = Mat4::Identity();
        yOffset(1, 3) = scale * offset;
        
        Mat4 zOffset = Mat4::Identity();
        zOffset(2, 3) = scale * offset;
        
        agl::Render::cube()
            ->transform(trf * xOffset)
            ->color(1, 0, 0)
            ->debug(true)
            ->scale(scale * 2 * offset, 0.03f, 0.03f)
            ->draw();
        
        agl::Render::cube()
            ->transform(trf * yOffset)
            ->color(0, 1, 0)
            ->debug(true)
            ->scale(0.03f, scale * 2 * offset, 0.03f)
            ->draw();

        agl::Render::cube()
            ->transform(trf * zOffset)
            ->color(0, 0, 1)
            ->debug(true)
            ->scale(0.03f, 0.03f, scale * 2 * offset)
            ->draw();
    }

    void draw_trj(int windowSize, Mat4 rt, vVec3 local, float r, float g, float b)
    {
        for (int i = 0; i < windowSize; ++i)
        {
            Mat4 root = rt;
            Vec3 local_trj_pos = local.at(i);
            Vec4 local_trj_pos_homo = Vec4::Ones();
            local_trj_pos_homo.block<3, 1>(0, 0) = local_trj_pos;

            Vec4 world_pos_homo = root * local_trj_pos_homo;

            agl::Render::sphere()
                ->position(world_pos_homo.head<3>())
                ->scale(0.05f)
                ->color(r, g, b)
                ->debug(true)
                ->draw();
        }
    }

    void draw_trj_directions(int windowSize, Mat4 rt, vVec3 local_pos, vVec3 local_dirs)
    {
        for (int i = 0; i < windowSize; ++i)
        {
            Mat4 root = rt;
            Vec3 local_trj_pos = local_pos.at(i);
            Vec4 local_trj_pos_homo = Vec4::Ones();
            local_trj_pos_homo.block<3, 1>(0, 0) = local_trj_pos;

            Vec4 world_pos_homo = root * local_trj_pos_homo;

            Vec3 local_trj_dir = local_dirs.at(i);
            Vec4 local_trj_dir_homo = Vec4::Zero();
            local_trj_dir_homo.block<3, 1>(0, 0) = local_trj_dir;

            Vec4 world_trj_dir_homo = root * local_trj_dir_homo;

            draw_direction(world_pos_homo.head(3), world_trj_dir_homo.head(3), 1.0f);
        }
    }
    
    
    void camera_fix(const Mat4 rt, const agl::spModel model, float sangle, Vec4 cam_local_pos = Vec4(0.0f, 2.0f, 4.0f, 1.0f))
    {
        Vec3 r_axis = Vec3::UnitY();

        Mat3 m3 = AAxis(sangle, r_axis).toRotationMatrix();
        Mat4 wtrf = model->root()->world_trf();
        wtrf.block<3, 3>(0, 0) = rt.block<3, 3>(0, 0);

        // wtrf.block<3, 3>(0, 0) = Mat3::Identity();

        wtrf.block<3, 3>(0, 0) = wtrf.block<3, 3>(0, 0) * m3;

        Vec4 cam_world_pos = wtrf * cam_local_pos;
        glm::vec3 cam_focus_pos(agl::to_glm((Vec4)wtrf.col(3)));

        glm::mat4 gtrf = a::gl::to_glm(wtrf);
        camera().set_focus(cam_focus_pos);
        glm::vec3 cam_pos_glm(agl::to_glm(cam_world_pos));
        camera().set_position(cam_pos_glm);

    }
    
    std::vector<Vec3> get_jnt_glb_pos(agl::spModel model, agl::Pose pose)
    {
        //variables
        std::vector<Vec3> jnt_prev_pos;    
        jnt_prev_pos.resize(joint_names.size());
        

        // set pose
        model->set_pose(pose);
        model->update_mesh();
        
        // get global position of joints
        for (int i = 0; i < joint_names.size(); ++i)
        {
            //get name of each joint
            std::string jnt = joint_names.at(i);
            jnt_prev_pos.at(i) =  model->joint(jnt)->world_pos();
        }
        
        return jnt_prev_pos;
    }
    
    agl::spModel model;
    std::vector<agl::Motion> motions;
    apfnn::ContactPhaseInfo info;

    apfnn::RootInfo root_info;
    std::vector<float> phase;

    std::vector<apfnn::PFNN_XY_FrameData> framesData;

    std::vector<std::string> joint_names;
    std::vector<int> sample_timing;
    const int windowSize = 12;

    apfnn::hparam param{205, 234, 32, 20, 10, 0.7};

    float ph;
    Mat4 rt;
    Tensor XTensor;
    Tensor YTensor;

    Tensor mean_stdev_t;
    Tensor ph_mstd;

    apfnn::PFNN_Y_Data reconY;
    agl::Pose reconPose;

    Tensor input;
    Tensor output;
    Tensor scaled_output;
    Tensor recon_output;
    
    apfnn::fcnet fcNet{nullptr};

    float z_vel = 0;
    float x_vel = 0;
    
    std::vector<Vec3> target_pos;
    std::vector<Vec3> original_pos;

    std::vector<Vec3> original_dirs;
    std::vector<Vec3> target_dirs;

    Vec3 target_dir = Vec3(0, 0, 0);

    float jnt_num;
    Tensor jnt_info; 

    bool isTrue = false;

    std::vector<Vec3> tmpp1;



    std::vector<Vec3> jnt_prev_pos;
    std::vector<Vec3> trj_glb_prev_pos; // size: 60
    std::vector<Vec3> trj_glb_prev_dir; // size: 60
    torch::Device device = torch::kCPU;
    Tensor temp1;
    void start() override
    {
        //** STEP1 variables setting
        const char *model_path = "../data/fbx/kmodel/model/kmodel.fbx";
        std::string const motion_path = "../data/fbx/kmodel/motion/";
        std::string const motion_name = "simple2";
        std::string const extension_name = ".fbx";
        std::string motion_full_path = motion_path + motion_name + extension_name;
        
        agl::FBX model_fbx(model_path);
        model = model_fbx.model();

        agl::FBX motion_fbx(motion_full_path);
        motions = motion_fbx.motion(model);

        agl::spJoint leftFoot = model->joint("LeftFoot");
        agl::spJoint rightFoot = model->joint("RightFoot");

        agl::spJoint LeftToeBase = model->joint("LeftToeBase");
        agl::spJoint LeftToe_End = model->joint("LeftToe_End");

        agl::spJoint LeftShoulder = model->joint("LeftShoulder");
        agl::spJoint RightShoulder = model->joint("RightShoulder");
        agl::spJoint Hips = model->joint("Hips");

        //** LeftHand & RightHand의 children을 제외한 나머지 joint들만 input joints로 사용한다.
        joint_names = apfnn::kmodel_jnt_names;

        //** sample_timing(sampled surrounding frames에 대한 정보)
        sample_timing = apfnn::sample_timings;

        std::cout << "joint_names.size(): " << joint_names.size() << std::endl;

        framesData.resize(motions.at(0).poses.size());

        //** STEP2 data setting (contact labeling, root info extracting, root info extracting)
        float foot_height = apfnn::approximate_standard_foot_height(LeftToeBase, LeftToe_End, leftFoot);
        apfnn::label_contact_info(info.left_contact, info.right_contact, model, motions.at(0),
                                  leftFoot, rightFoot, foot_height, 0.06f, 1.13f, (1.0f / 60.0f));

        apfnn::filter_contact_info(info.left_contact, info.right_contact, 1, 1, 4);

        apfnn::label_phase_info(info.left_contact, info.right_contact, phase, 30);

        apfnn::get_root_info(root_info.root_world_trf, model, motions.at(0), LeftShoulder, RightShoulder, Hips);


        //** STEP3 get data for each frame
        int min_frame = -sample_timing.at(0);
        int max_frame = (framesData.size() - sample_timing.at(windowSize - 1));

        for (int i = min_frame; i < max_frame; ++i)
        {
            framesData.at(i) = apfnn::get_frame_data(i, model, motions.at(0).poses, 
                                                   root_info.root_world_trf, phase, 
                                                   joint_names, sample_timing, 
                                                   info.left_contact.at(i), info.right_contact.at(i));
            if(i % 1000 == 0)
            {
                std::cout << i << " th XYframeData setting done" << std::endl;
            }
        }

        std::cout << "************* XYframeData Setting Done *************\n" << std::endl;

        Tensor YTensor;
        {
            torch::load(XTensor, "./ptfiles/tensors/" + motion_name + "/final_Xtensor_fc.pt");
            torch::load(YTensor, "./ptfiles/tensors/" + motion_name + "/final_Ytensor_fc.pt");
            torch::load(mean_stdev_t, "./ptfiles/tensors/" + motion_name + "/mean_stdev_fc.pt");
            torch::load(ph_mstd, "./ptfiles/tensors/" + motion_name + "/ph_mstd_fc.pt");

            std::cout << "final_XTensor.sizes(): " << XTensor.sizes() << std::endl; // [14157, 234]
            // std::cout << "final_YTensor.sizes(): " << YTensor.sizes() << std::endl; // [14157, 279]
            std::cout << "mean_stdev_t.sizes(): " << mean_stdev_t.sizes() << std::endl; // [14157, 279]
            std::cout << "*************Restoration Done*************\n\n" << std::endl;
        }

        //** network setting
        fcNet = apfnn::fcnet(param.inputSize, param.outputSize, false, param.dprob);
        
        // since it is training network, make sure that the network is on training
        if (fcNet->is_train == true)
        {
            fcNet->is_train = false;
        }
        torch::load(fcNet, "./ptfiles/fcNet_pts/" + motion_name + "/fcNet-checkpoint.pt");
        torch::load(mean_stdev_t, "./ptfiles/tensors/" + motion_name + "/mean_stdev_fc.pt");
        //** device setting
        if (torch::cuda::is_available())
        {
            std::cout << "CUDA is available! Training on GPU." << std::endl;
            device = torch::kCUDA;
       
            fcNet->to(device);
            YTensor = YTensor.to(device);
            mean_stdev_t = mean_stdev_t.to(device);
        }

        int idx = 101;
        int scaled_idx = idx - (min_frame + 1);

        framesData.at(idx) = apfnn::get_frame_data(idx, model, motions.at(0).poses,
                                                   root_info.root_world_trf, phase,
                                                   joint_names, sample_timing,
                                                   info.left_contact.at(idx), info.right_contact.at(idx));

        trj_glb_prev_pos.resize(-sample_timing.at(0));
        trj_glb_prev_dir.resize(-sample_timing.at(0));

        for (int i = 0; i < trj_glb_prev_pos.size(); ++i)
        {
            int frame_idx = idx + sample_timing.at(0) + i;
            Mat4 frame_trf = root_info.root_world_trf.at(frame_idx);
            
            //** get root_global_trj_position
            trj_glb_prev_pos.at(i) = frame_trf.block<3, 1>(0, 3);

            //** get root_global_trj_direction
            trj_glb_prev_dir.at(i) = frame_trf.block<3, 1>(0, 2);
        }

        // get joint previous global position
        jnt_prev_pos = get_jnt_glb_pos(model, motions.at(0).poses.at(idx));
        
        ph = phase.at(idx);
        rt = root_info.root_world_trf.at(idx);
        input = XTensor[0, scaled_idx].to(device);

        output = fcNet->forward(input);

        if(output.dim() == 2)
        {
            output.squeeze_(0);
        }

        std::cout << "phase: " << ph * 180.0f / (M_PI) << std::endl;

        // scale up and denormalize output
        scaled_output = apfnn::scale_tensor(output, torch::range(4 * windowSize, 
                        param.outputSize - 7, torch::kInt64), 10.0f);
        recon_output = apfnn::denormalize_tensor(scaled_output, mean_stdev_t);

        // make y_data from Y_Tensor for reconstruction
        reconY = apfnn::parse_Y_tensor_to_y_data(model, ph, rt,
                                                 recon_output,
                                                 windowSize, joint_names);

        ph = reconY.phase;
        rt = reconY.world_root_trf;
        
        // reconstruct pose
        reconPose = apfnn::reconstruct_pose(model, reconY, joint_names);
        
        // set pose
        model->set_pose(reconPose);
        model->update_mesh();
        Tensor gtruth = YTensor[0, scaled_idx];

        // loss
        Tensor loss = torch::mse_loss(output, gtruth);

        // std::cout << "loss is: " << output - gtruth << std::endl;
        std::cout << "loss is: " << loss << std::endl;
        
        jnt_num = joint_names.size();
        
        jnt_info = apfnn::jntInfo_fromRecon(model, reconPose, rt, joint_names, jnt_prev_pos).to(device);
        
        temp1 = jnt_info;
        
        jnt_info = apfnn::normalize_tensor_with_stdev(jnt_info, mean_stdev_t,
                                                      torch::range(4 * windowSize, 4 * windowSize +  6 * jnt_num - 1,
                                                                   torch::kInt64).to(device));
        jnt_info = apfnn::scale_tensor(jnt_info, torch::range(0, 6 * jnt_num - 1, torch::kInt64), 0.1f);

        jnt_prev_pos = get_jnt_glb_pos(model, reconPose);

    }

    void update() override
    {
        if(true)
        {

        camera_fix(reconY.world_root_trf, model, 1.5f * M_PI, Vec4(0, 1, 4, 1));

        // 1. use previous trajectories which has been recorded continuously 
        // erase the oldest trajectory pos and dir            
        trj_glb_prev_pos.erase(trj_glb_prev_pos.begin());
        trj_glb_prev_dir.erase(trj_glb_prev_dir.begin());

        trj_glb_prev_pos.push_back(rt.block<3, 1>(0, 3));
        trj_glb_prev_dir.push_back(rt.block<3, 1>(0, 2));

        Tensor trj_prev_lposT;
        Tensor trj_prev_ldirT;
        vVec3 trj_prev_lposV;
        vVec3 trj_prev_ldirV;

        // 2. get TENSOR DATA - trjInfo for previous trajectories
        apfnn::prev_trajInfo_fromRecon(trj_prev_lposT, trj_prev_ldirT,
                                       trj_prev_lposV, trj_prev_ldirV,
                                       trj_glb_prev_pos, trj_glb_prev_dir, 
                                       windowSize, rt, sample_timing);
                  
        trj_prev_lposT = apfnn::normalize_tensor_with_stdev(trj_prev_lposT, mean_stdev_t,
                                                            torch::range(0, windowSize - 1, torch::kInt64).to(device));

        trj_prev_ldirT = apfnn::normalize_tensor_with_stdev(trj_prev_ldirT, mean_stdev_t,
                                                            torch::range(2 * windowSize, 3 * windowSize - 1, torch::kInt64).to(device));


        // 3. get TENSOR DATA; user control - trjInfo for future trajectories
        Tensor trj_lat_lposT;
        Tensor trj_lat_ldirT;
        apfnn::latter_trajInfo_fromRecon(trj_lat_lposT, trj_lat_ldirT,
                                         reconY.latter_trj_lpos, 
                                         reconY.latter_trj_ldir, windowSize);

        trj_lat_lposT = apfnn::normalize_tensor_with_stdev(trj_lat_lposT, mean_stdev_t,
                                                            torch::range(windowSize, 2 * windowSize - 1, torch::kInt64).to(device));

        trj_lat_ldirT = apfnn::normalize_tensor_with_stdev(trj_lat_ldirT, mean_stdev_t,
                                                            torch::range(3 * windowSize, 4 * windowSize - 1, torch::kInt64).to(device));

        // used before blending trj
        // Tensor trj_lat_lposT = torch::index_select(output, 0, torch::range(windowSize, 2 * windowSize - 1, torch::kInt64).to(device));
        // Tensor trj_lat_ldirT = torch::index_select(output, 0, torch::range(3 * windowSize, 4 * windowSize - 1, torch::kInt64).to(device));
        Tensor trj_height = torch::zeros(3 * windowSize).to(device);
        trj_height = trj_height.to(device);

        Tensor tmp = torch::tensor({ph}).to(device);
        tmp = apfnn::normalize_tensor_with_stdev(tmp, ph_mstd, torch::range(0, 0, torch::kInt64).to(device));

        input = torch::cat({trj_prev_lposT, trj_lat_lposT, trj_prev_ldirT, trj_lat_ldirT, trj_height, jnt_info, tmp});
        // input = torch::cat({trj_prev_lposT, trj_lat_lposT, trj_prev_ldirT, trj_lat_ldirT, trj_height, jnt_info});

        // get output from FC layer
        output = fcNet->forward(input);
        //! 이부분도 고쳐야함.(output.dim() == 2임.) denormalize(param의 dim == 1이어야함)에서 충돌
        if(output.dim() == 2)
        {
            output.squeeze_(0);
        }

        // scale up and denormalize output
        scaled_output = apfnn::scale_tensor(output, torch::range(4 * windowSize, param.outputSize - 7, 
                                            at::device(device).dtype(torch::kInt64)), 10.0f);

        recon_output = apfnn::denormalize_tensor(scaled_output, mean_stdev_t);

        // make y_data from Y_Tensor for reconstruction
        reconY = apfnn::parse_Y_tensor_to_y_data(model, ph, rt, recon_output, windowSize, joint_names);
        
        
        // reconstruct pose
        reconPose = apfnn::reconstruct_pose(model, reconY, joint_names);
        ph = reconY.phase;
        rt = reconY.world_root_trf;
        target_dir = Vec3(x_vel, 0, z_vel);
        std::cout << "phase: " << ph * 180.0f / (M_PI) << std::endl;

        // set pose
        model->set_pose(reconPose);
        model->update_mesh();

        jnt_info = apfnn::jntInfo_fromRecon(model, reconPose, rt, joint_names, jnt_prev_pos);
        temp1 = jnt_info;

        jnt_info = apfnn::normalize_tensor_with_stdev(jnt_info, mean_stdev_t,
                                                      torch::range(4 * windowSize, 4 * windowSize +  6 * jnt_num - 1,
                                                                   torch::kInt64).to(device));
        jnt_info = apfnn::scale_tensor(jnt_info, torch::range(0, 6 * jnt_num - 1, torch::kInt64), 0.1f);

        jnt_prev_pos =  get_jnt_glb_pos(model, reconPose);

        for (int i = 0; i < windowSize / 2; ++i)
        {
            reconY.latter_trj_lpos.at(i) = trj_prev_lposV.at(i);
            reconY.latter_trj_ldir.at(i) = trj_prev_ldirV.at(i);
        }

        //** trajectory blending
        // original trajectories
        original_pos = reconY.latter_trj_lpos;

        // original trajectories directions
        original_dirs = reconY.latter_trj_ldir;

        // blended trajectories positions
        reconY.latter_trj_lpos = apfnn::blend_traj_poss(rt, reconY.latter_trj_lpos, 0.5, target_dir, windowSize);

        // blended trajectories directions
        reconY.latter_trj_ldir = apfnn::blend_traj_dirs(rt, reconY.latter_trj_ldir, 0.5, target_dir, windowSize);
        isTrue = false;
      
        }
    }
        
    
    void render() override
    {
        agl::Render::plane()
            ->scale(200.0f)
            ->color(0.15f, 0.15f, 0.15f)
            ->floor_grid(true)
            ->draw();

        agl::Render::model(model)
            ->draw();
    }
        

    void render_xray() override
    {
        agl::Render::skeleton(model)
            ->color(0.9, 0.9, 0)
            ->draw();

        //** DRAW world_root_trf
        draw_local_coord(rt, 1.0f);


        // //** target Positions (User Control) - Yellow
        // draw_trj(windowSize, rt, target_pos, 1, 1, 0);

        // //** target directions (User control)
        // draw_trj_directions(windowSize, rt, target_pos, target_dirs);

        //** original Positions - Red
        draw_trj(windowSize, rt, original_pos, 1, 0, 0);

        // //** original directions
        // draw_trj_directions(windowSize, rt, original_pos, original_dirs);

        // //** Blended TRAJECTORIES_POSITIONS - Pink
        // draw_trj(windowSize, rt, reconY.latter_trj_lpos, 1, 0, 1);        

        // //** TRAJECTORIES_DIRECTIONS
        // draw_trj_directions(windowSize, rt, reconY.latter_trj_lpos, reconY.latter_trj_ldir);

        //** Blended TRAJECTORIES_POSITIONS - Pink
        draw_trj(windowSize, rt, reconY.latter_trj_lpos, 1, 0, 1);        

        //** TRAJECTORIES_DIRECTIONS
        draw_trj_directions(windowSize, rt, reconY.latter_trj_lpos, reconY.latter_trj_ldir);
        
        for (int i = 0; i < 20; ++i)
        {
            //** POSITION
            Vec3 local_pos = reconY.current_jnt_lpos.at(i);
            Vec4 local_pos_homo = Vec4::Ones();
            local_pos_homo.block<3, 1>(0, 0) = local_pos;

            Vec4 world_pos_homo = reconY.world_root_trf * local_pos_homo;
            Vec3 world_pos = world_pos_homo.block<3, 1>(0, 0);
            Vec3 real_world_pos = model->joint(joint_names.at(i))->world_pos();
            // std::cout << "difference: " << world_pos - real_world_pos << std::endl;
            // agl::Render::sphere()
            //     ->position(tmpp1.at(i))
            //     ->scale(0.07f)
            //     ->color(0, 1, 1)
            //     ->debug(true)
            //     ->draw();

            // //** ORIENTATION
            Vec3 temp = reconY.current_jnt_laaxis.at(i);
            float angle = temp.norm();
            Vec3 axis = temp / angle;
            AAxis local_aaxis = AAxis(angle, axis);
            Quat local_quat(local_aaxis);

            // //** VELOCITY
            Vec3 local_vel = reconY.current_jnt_lvel.at(i);
            local_vel = Vec3(temp1[3 * jnt_num + 3 * i].item<float>(), 
                                  temp1[3 * jnt_num + 3 * i + 1].item<float>(), 
                                  temp1[3 * jnt_num + 3 * i + 2].item<float>());
            Vec4 local_vel_homo = Vec4::Zero();
            local_vel_homo.block<3, 1>(0, 0) = local_vel;
            Vec4 world_vel_homo = rt * local_vel_homo;

            Vec4 debug_pos = world_pos_homo;

            // Tensor temp = torch::index_select(jnt_info, 0, torch::range(3 * jnt_num, 6 * jnt_num - 1, torch::kInt64).to(device));
            // std::cout << temp << std::endl;
            // std::cout << "**********"<<std::endl;  
            // draw_velocity(real_world_pos, world_vel_homo.head<3>(), 1.0f);
        }

    }

    void key_callback(char key, int action)
    {

        if (key == 'w' && action == GLFW_PRESS)
        {
            z_vel += 1;
        }
        else if (key == 'w' && action == GLFW_RELEASE)
        {
            z_vel = 0;
        }

        if (key == 's' && action == GLFW_PRESS)
        {
            z_vel -= 1;
        }
        else if (key == 's' && action == GLFW_RELEASE)
        {
            z_vel = 0;
        }

        if (key == 'a' && action == GLFW_PRESS)
        {
            x_vel += 1;
            z_vel += 1;
        }
        
        else if (key == 'a' && action == GLFW_RELEASE)
        {
            x_vel = 0;
            z_vel -= 1;
        }

        if (key == 'd' && action == GLFW_PRESS)
        {
            x_vel -= 1;
            z_vel += 1;
        }
        else if (key == 'd' && action == GLFW_RELEASE)
        {
            x_vel = 0;
            z_vel -= 1;
        }
        
        if (action != GLFW_PRESS)
            return;

        if(key == 'p')
        {
            isTrue = true;
        }
    }
};

int main(int argc, char* argv[])
{
    MyApp app;
    agl::AppManager::start(&app);
    return 0;
}