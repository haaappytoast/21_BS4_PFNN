#include <aPfnn.h>
#include <iostream>

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

    agl::spModel model;
    std::vector<agl::Motion> motions;
    apfnn::ContactPhaseInfo info;

    apfnn::RootInfo root_info;
    std::vector<float> phase;

    std::vector<apfnn::PFNN_XY_FrameData> framesData;

    apfnn::PFNN_XY_FrameData frame_data;

    std::vector<std::string> joint_names;



    std::vector<int> sample_timing;
    const int windowSize = 12;

    int update_freq = 1;

    int frame =  101 * update_freq;
    int pidx;

    void start() override
    {
        const char* model_path  = "../data/fbx/kmodel/model/kmodel.fbx";
        const char* motion_path = "../data/fbx/kmodel/motion/simple2.fbx";

        //** variables setting
        agl::FBX model_fbx(model_path);
        model = model_fbx.model();
        
        agl::FBX motion_fbx(motion_path);
        motions = motion_fbx.motion(model);

        agl::spJoint leftFoot = model->joint("LeftFoot");
        agl::spJoint rightFoot = model->joint("RightFoot");

        agl::spJoint LeftToeBase = model->joint("LeftToeBase");
        agl::spJoint LeftToe_End = model->joint("LeftToe_End");
        agl::spJoint RightToe_End = model->joint("RightToe_End");

        agl::spJoint LeftShoulder = model->joint("LeftShoulder");
        agl::spJoint RightShoulder = model->joint("RightShoulder");
        agl::spJoint Hips = model->joint("Hips");

        //** LeftHand & RightHand의 children을 제외한 나머지 joint들만 input joints로 사용한다.
        joint_names = apfnn::kmodel_jnt_names;

        //** sample_timing(sampled surrounding frames에 대한 정보)
        sample_timing = apfnn::sample_timings;

        // for (auto jnt : model->joints())
        // {
        //     std::cout << jnt->name() << std::endl;
        // }

        std::cout << "joint_names.size(): " << joint_names.size() << std::endl;

        framesData.resize(motions.at(0).poses.size());

        //** data setting (contact labeling, root info extracting, Tensor정보 얻기)
        float foot_height = apfnn::approximate_standard_foot_height(LeftToeBase, LeftToe_End, leftFoot);
        apfnn::label_contact_info(info.left_contact, info.right_contact, model, motions.at(0), 
                                leftFoot, rightFoot, foot_height, 0.06f, 1.13f, (1.0f / 60.0f));

        apfnn::filter_contact_info(info.left_contact, info.right_contact, 1, 1, 4);
        
        apfnn::label_phase_info(info.left_contact, info.right_contact, phase, 30);

       apfnn::get_root_info(root_info.root_world_trf, model, motions.at(0), LeftShoulder, RightShoulder, Hips);

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

        std::vector<Tensor> X_tensor_lists;
        std::vector<Tensor> Y_tensor_lists;
        Tensor XTensor;
        Tensor YTensor;

        if (0)
        {
            for (int i = min_frame + 1; i < max_frame - 1; ++i)
            {
                Tensor x_i = apfnn::to_X_tensor(framesData.at(i - 1), framesData.at(i));
                Tensor y_i = apfnn::to_Y_tensor(framesData.at(i), framesData.at(i + 1));

                X_tensor_lists.push_back(x_i);
                Y_tensor_lists.push_back(y_i);

                if ((i % max_frame-1) == 0)
                {
                    std::cout << i << " th Tensor Data setting done" << std::endl;
                }
            }
            //** Change it into Large Tensor
            XTensor = torch::vstack(X_tensor_lists);
            YTensor = torch::vstack(Y_tensor_lists);
            std::cout << "XTensor.sizes(): " << XTensor.sizes() << std::endl; // [14157, 234]
            std::cout << "YTensor.sizes(): " << YTensor.sizes() << std::endl; // [14157, 279]
            std::cout << "\n************* Data Setting Done *************\n"
                      << std::endl;

            torch::save(XTensor, "./ptfiles/tensors/Xtensor.pt");
            torch::save(YTensor, "./ptfiles/tensors/Ytensor.pt");
        }
        else
        {
            torch::load(XTensor, "./ptfiles/tensors/Xtensor.pt");
            torch::load(YTensor, "./ptfiles/tensors/Ytensor.pt");
            std::cout << "XTensor.sizes(): " << XTensor.sizes() << std::endl; // [14157, 234]
            std::cout << "YTensor.sizes(): " << YTensor.sizes() << std::endl; // [14157, 279]
        }

    // {
    //     //! 지워야해!
    //     Tensor y_i = apfnn::to_Y_tensor(framesData.at(frame), framesData.at(frame + 1));
    //     int pidx = frame / update_freq;
    //     const auto &pose = motions.at(0).poses.at(pidx);
    //     model->set_pose(pose);

    //     std::cout << model->joint("Hips")->world_rot() << std::endl;
    //     std::cout << model->joint("Hips")->local_rot() << std::endl;
    //     apfnn::PFNN_Y_Data y_data =
    //         apfnn::Parse_Y_tensor_to_y_data(model, phase.at(frame-1), root_info.root_world_trf.at(frame - 1), y_i, windowSize, joint_names);

    //     agl::Pose dfdf = apfnn::reconstruct_pose(model, y_data);
    //     for (int i = 0; i < dfdf.local_rotations.size(); ++i)
    //     {
    //         std::cout << "diff\n"
    //                   << (dfdf.local_rotations.at(i)).toRotationMatrix() - (motions.at(0).poses.at(frame).local_rotations.at(i)).toRotationMatrix() << std::endl;
    //     }
    //     Vec3 d = dfdf.root_position - (motions.at(0).poses.at(frame).root_position);
    //     std::cout << "position diff:\n"
    //               << d << std::endl;
    //     std::cout << phase.at(frame) - y_data.phase << std::endl;
    //     exit(0);
    // }

       // example
        if(0)
        {
            Quat q;
            AAxis aaxis(q);
            float angle = aaxis.angle(); // scalar
            Vec3 axis = aaxis.axis();    // unit vector
            Vec3 aaxis_in_3d = angle * axis;
        }

    }

    void update() override
    {

        const auto& motion = motions.at(0);
        pidx = frame / update_freq;
        const auto& pose = motion.poses.at(pidx);
        model->set_pose(pose);
        frame_data = framesData.at(pidx);
        model->update_mesh();

        frame = (frame + 1) % (int)(motion.poses.size() * update_freq);
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
        
        bool is_right_contact = info.right_contact.at(pidx);
        bool is_left_contact = info.left_contact.at(pidx);

        Vec3 rightFoot = model->joint("RightFoot")->world_pos();
        Vec3 leftFoot = model->joint("LeftFoot")->world_pos();
        Vec3 Hips = model->joint("Hips")->world_pos();

        {    
            if(is_right_contact)
            {
                agl::Render::sphere()
                    ->position(rightFoot)
                    ->scale(0.18f)
                    ->color(1, 0, 0)
                    ->debug(true)
                    ->draw();
            }

            if(is_left_contact)
            {
                agl::Render::sphere()
                    ->position(leftFoot)
                    ->scale(0.18f)
                    ->color(0, 1, 0)
                    ->debug(true)
                    ->draw();
            }

            float angle = 1* M_PI/2;
            Mat3 rotation = root_info.root_world_trf.at(pidx).block<3, 3>(0, 0);
            Vec3 ph = rotation * Vec3{0.3f * -cos(phase.at(pidx)), 0, -0.3f * sin(phase.at(pidx))};
            
            float weight = abs(sin(0.5f * phase.at(pidx)));
            Vec3 clr = weight * Vec3(0, 1, 0) + (1.0f - weight) * Vec3(1, 0, 0);
            agl::Render::sphere()
                ->position(Hips + ph)
                ->scale(0.2f)
                ->color(clr)
                ->debug(true)
                ->draw();
            }
    }
        

    void render_xray() override
    {
        //draw_velocity(Vec3::Zero(), Vec3(1, 2, 3), 1.0f);


        agl::Render::skeleton(model)
            ->color(0.9, 0.9, 0)
            ->draw();

        //draw_local_coord(model->joint("Hips")->world_trf());
        draw_local_coord(root_info.root_world_trf.at(pidx), 1.5f);

            // How To Draw
            {
                int noj = frame_data.jnts_local_positions.size();
                int windowSize = frame_data.local_trj_positions.size();

                for (int i = 0; i < noj; ++i)
                {
                    //** POSITION
                    Vec3 local_pos = frame_data.jnts_local_positions.at(i);
                    Vec4 local_pos_homo = Vec4::Ones();
                    local_pos_homo.block<3, 1>(0, 0) = local_pos;

                    Vec4 world_pos_homo = frame_data.world_root_trf * local_pos_homo;
                    Vec3 world_pos = model->joint(joint_names.at(i))->world_pos();

                     agl::Render::sphere()
                        ->position(world_pos)
                        ->scale(0.07f)
                        ->color(1, 0, 0)
                        ->debug(true)
                        ->draw();

                    //** ORIENTATION
                    Vec3 temp = frame_data.jnts_local_aaxis.at(i);
                    float angle = temp.norm();
                    Vec3 axis = temp / angle;
                    AAxis local_aaxis = AAxis(angle, axis);
                    Quat local_quat(local_aaxis);

                    // //** VELOCITY
                    Vec3 local_vel = frame_data.jnts_local_velocity.at(i);
                    Vec4 local_vel_homo = Vec4::Zero();
                    local_vel_homo.block<3, 1>(0, 0) = local_vel;
                    Vec4 world_vel_homo = frame_data.world_root_trf * local_vel_homo;

                    Vec4 debug_pos = world_pos_homo;

                    draw_velocity(debug_pos.head<3>(), world_vel_homo.head<3>(), 1.0f);
                }

                //** ROOT_LOCAL_VELOCITY
                Vec3 root_local_vel = frame_data.root_local_velocity;
                Vec4 root_local_vel_homo = Vec4::Zero();
                root_local_vel_homo.block<3, 1>(0, 0) = root_local_vel;
                Vec4 root_world_vel_homo = frame_data.world_root_trf * root_local_vel_homo;

                Vec3 root_pos = frame_data.world_root_trf.col(3).head(3);
                draw_velocity(root_pos, root_world_vel_homo.head<3>(), 1.0f);

                //** TRAJECTORIES_POSITIONS
                for (int i = 0; i < windowSize; ++i)
                {
                    int frame_idx = pidx + sample_timing.at(i);
                    Mat4 frame_trf = root_info.root_world_trf.at(frame_idx);
                    Vec3 frame_pos = frame_trf.block<3, 1>(0, 3);

                    Vec3 local_pos = frame_data.local_trj_positions.at(i);
                    Vec4 local_pos_homo = Vec4::Ones();
                    local_pos_homo.block<3, 1>(0, 0) = local_pos;

                    Vec4 world_pos_homo = frame_data.world_root_trf * local_pos_homo;

                    agl::Render::sphere()
                        ->position(world_pos_homo.head<3>())
                        ->scale(0.05f)
                        ->color(1, 0, 1)
                        ->debug(true)
                        ->draw();

                }

                //** TRAJECTORIES_DIRECTIONS
                for (int i = 0; i < windowSize; ++i)
                {
                    int frame_idx = pidx + sample_timing.at(i);
                    Mat4 frame_trf = root_info.root_world_trf.at(frame_idx);

                    Vec3 local_trj_dir = frame_data.local_trj_directions.at(i);
                    Vec4 local_trj_dir_homo = Vec4::Zero();
                    local_trj_dir_homo.block<3, 1>(0, 0) = local_trj_dir;

                    Vec4 world_trj_dir_homo = frame_data.world_root_trf * local_trj_dir_homo;

                   draw_direction(frame_trf.col(3).head(3), world_trj_dir_homo.head(3), 1.0f);

                }
           }
    }

    void key_callback(char key, int action)
    {
        if(action != GLFW_PRESS)
            return;
        
        if(key == 's')
        {
            frame += 60;
            std::cout << "frame" << (int) frame/update_freq << std::endl;
        }
        if(key == 'a')
        {
            frame -= 60;
            std::cout << "frame" << (int) frame/update_freq << std::endl;
        }
        
        //frame = (frame) % (int)(motion.poses.size() * update_freq);
    }
};

int main(int argc, char* argv[])
{
    MyApp app;
    agl::AppManager::start(&app);
    return 0;
}