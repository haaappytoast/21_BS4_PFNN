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
    
    
    void camera_fix(const Mat3 rt, const agl::spModel model, float sangle, Vec4 cam_local_pos = Vec4(0.0f, 2.0f, 4.0f, 1.0f))
    {
        Vec3 r_axis = Vec3::UnitY();

        Mat3 m3 = AAxis(sangle, r_axis).toRotationMatrix();
        Mat4 wtrf = model->root()->world_trf();
        //wtrf.block<3, 3>(0, 0) = m3;
        // wtrf.block<3, 3>(0, 0) = Mat3::Identity();
        wtrf.block<3, 3>(0, 0) = rt;

        wtrf(1, 3) = 0.5f;

        Vec4 cam_world_pos = wtrf * cam_local_pos;

        glm::vec3 cam_focus_pos(agl::to_glm((Vec4)wtrf.col(3)));

        glm::mat4 gtrf = a::gl::to_glm(wtrf);
        camera().set_focus(cam_focus_pos);
        glm::vec3 cam_pos_glm(agl::to_glm(cam_world_pos));
        camera().set_position(cam_pos_glm);

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

    apfnn::hparam param{234, 279, 32, 20, 50, 0.3};

    float ph;
    Mat4 rt;
    Tensor XTensor;
    Tensor mean_stdev_t;

    apfnn::PFNN_Y_Data reconY;
    agl::Pose reconPose;

    Tensor input;
    Tensor output;
    Tensor scaled_output;
    Tensor recon_output;
    
    apfnn::fcnet fcNet{nullptr};

    float z_vel = 0;
    float x_vel = 0;
    
    std::vector<Tensor> tensor_lists;
    std::vector<Tensor> tensor_lists2;
    
    std::vector<Vec3> target_pos;
    std::vector<Vec3> original_pos;

    std::vector<Vec3> original_dirs;
    std::vector<Vec3> target_dirs;

    const Tensor pos_idx = torch::range(0, 2 * windowSize - 1, torch::kInt64);
    const Tensor dir_idx = torch::range(2 * windowSize, 4 * windowSize - 1, torch::kInt64);

    Vec3 target_dir = Vec3(0, 0, 0);

    int count = 0;

    Tensor example[4];

    void start() override
    {
    example[0] = torch::tensor({{1,2},{3,4},{5,6}, {1,2}, {3,4}, {5,6}});


        // Tensor a= torch::tensor({{1,2},{3,4},{5,6}, {1,2}, {3,4}, {5,6}});
        // Tensor index_selected = torch::index_select(mean_stdev_t, 0, pos_idx);
        // Tensor mean = torch::squeeze(torch::narrow(index_selected, 1, 0, 1));
        // std::cout << index_selected << std::endl;
        // std::cout << "***" <<std::endl;
        // std::cout << mean << std::endl;
        // exit(0);

        //** STEP1 variables setting
        const char *model_path = "../data/fbx/kmodel/model/kmodel.fbx";
        const char *motion_path = "../data/fbx/kmodel/motion/ubi_run1_subject5.fbx";

        agl::FBX model_fbx(model_path);
        model = model_fbx.model();

        agl::FBX motion_fbx(motion_path);
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

        //** prepare tensor data
        torch::load(XTensor, "./ptfiles/fcNet_pts/final_Xtensor.pt");

        //** network setting
        fcNet = apfnn::fcnet(param.inputSize, param.outputSize, false, param.dprob);
        
        torch::load(fcNet, "./ptfiles/fcNet_pts/fcNet-checkpoint.pt");
        torch::load(mean_stdev_t, "./ptfiles/fcNet_pts/mean_stdev.pt");

        //** device setting
        torch::Device device = torch::kCPU;

        if (torch::cuda::is_available())
        {
            // std::cout << "CUDA is available! Training on GPU." << std::endl;
            // device = torch::kCUDA;
        }
        fcNet->to(device);

        int idx = 1000;
        ph = phase.at(idx);
        rt = root_info.root_world_trf.at(idx);

        input = XTensor[0, idx];
        output = fcNet->forward(input);

        // scale up and denormalize output
        scaled_output = apfnn::scale_tensor(output, torch::range(4 * windowSize, param.outputSize - 7, torch::kInt64), 10.0f);
        recon_output = apfnn::denormalize_tensor(scaled_output, mean_stdev_t);

        // make y_data from Y_Tensor for reconstruction
        reconY = apfnn::parse_Y_tensor_to_y_data(model, ph, rt,
                                                 recon_output,
                                                 windowSize, joint_names);
        // reconstruct pose
        reconPose = apfnn::reconstruct_pose(model, reconY, joint_names);
        // set pose
        model->set_pose(reconPose);
        model->update_mesh();

        tensor_lists.resize(windowSize);
        tensor_lists2.resize(windowSize);
    }

    void update() override
    {
        if(count != 0)
        {

        }

        else{
            count +=1;
            std::cout << target_dir << std::endl;
            std::cout << "***" << std::endl;
        //camera_fix(rt.block<3, 3>(0, 0), model, 3 * M_PI_2, Vec4(0.0f, 2.0f, -5.0f, 1.0f));

        // recursively get XTensor from YTensor
        input = apfnn::to_X_tensor_from_YTensor(output, windowSize, joint_names.size());
        
        // user control
        for (int i = 0; i < windowSize; ++i)
        {
            Tensor pos_t = akin::vec_to_tensor(Vec2(reconY.latter_trj_lpos.at(i).block<1, 1>(0, 0).value(),
                                                  reconY.latter_trj_lpos.at(i).block<1, 1>(2, 0).value()));
            tensor_lists.at(i) = pos_t;

            Tensor dir_t = akin::vec_to_tensor(Vec2(reconY.latter_trj_ldir.at(i).block<1, 1>(0, 0).value(),
                                                  reconY.latter_trj_ldir.at(i).block<1, 1>(2, 0).value()));
            tensor_lists2.at(i) = dir_t;
        }

        Tensor trj_pos = torch::cat(tensor_lists, 0);
        Tensor trj_dir = torch::cat(tensor_lists2, 0);
    
        // normalize tensor
        trj_pos = apfnn::normalize_tensor_with_stdev(trj_pos, mean_stdev_t, pos_idx);
        trj_dir = apfnn::normalize_tensor_with_stdev(trj_dir, mean_stdev_t, dir_idx);

        Tensor other_input = input.split_with_sizes({4 * windowSize, input.size(0) - 4 * windowSize}).at(1);

        input = torch::cat({trj_pos, trj_dir, other_input});

        // get output from FC layer
        output = fcNet->forward(input);

        // scale up and denormalize output
        scaled_output = apfnn::scale_tensor(output, torch::range(4 * windowSize, param.outputSize - 7, torch::kInt64), 10.0f);
        recon_output = apfnn::denormalize_tensor(scaled_output, mean_stdev_t);

        // make y_data from Y_Tensor for reconstruction
        reconY = apfnn::parse_Y_tensor_to_y_data(model, ph, rt, recon_output, windowSize, joint_names);

        // reconstruct pose
        reconPose = apfnn::reconstruct_pose(model, reconY, joint_names);

        // set pose
        model->set_pose(reconPose);
        model->update_mesh();

        ph = reconY.phase;
        rt = reconY.world_root_trf;

        target_dir = Vec3(x_vel, 0, z_vel);

        // original trajectories
        original_pos = reconY.latter_trj_lpos;

        // original trajectories directions
        original_dirs = reconY.latter_trj_ldir;

        // target position - bias = 0
        target_pos = apfnn::blend_traj_poss(rt, reconY.latter_trj_lpos, 0, target_dir, windowSize);

        // target velocities - bias = 0
        target_dirs =  apfnn::blend_traj_dirs(rt, reconY.latter_trj_ldir, 0, target_dir, windowSize);

        // blended trajectories positions
        reconY.latter_trj_lpos = apfnn::blend_traj_poss(rt, reconY.latter_trj_lpos, 0.5, target_dir, windowSize);

        // blended trajectories directions
        reconY.latter_trj_ldir = apfnn::blend_traj_dirs(rt, reconY.latter_trj_ldir, 0.5, target_dir, windowSize);
        
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


        //** target Positions (User Control) - Yellow
        draw_trj(windowSize, rt, target_pos, 1, 1, 0);

        //** target directions (User control)
        draw_trj_directions(windowSize, rt, target_pos, target_dirs);

        //** original Positions - Red
        draw_trj(windowSize, rt, original_pos, 1, 0, 0);

        //** original directions
        draw_trj_directions(windowSize, rt, original_pos, original_dirs);

        //** Blended TRAJECTORIES_POSITIONS - Pink
        draw_trj(windowSize, rt, reconY.latter_trj_lpos, 1, 0, 1);        

        //** TRAJECTORIES_DIRECTIONS
        draw_trj_directions(windowSize, rt, reconY.latter_trj_lpos, reconY.latter_trj_ldir);


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

        if (key == 'r' && action == GLFW_PRESS)
        {
            count = 0;
        }
        
        if (action != GLFW_PRESS)
            return;
        
    }
};

int main(int argc, char* argv[])
{
    MyApp app;
    agl::AppManager::start(&app);
    return 0;
}