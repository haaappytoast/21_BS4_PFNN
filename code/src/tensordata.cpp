#include "aPFNN/tensordata.h"
#include <iostream>

namespace a::pfnn
{
        /**
     * v3: 차원을 줄이고 싶은 vec3 
     * r0: output vec2의 row(0)에 넣어줄 v3의 row index
     * r1: output vec2의 row(1)에 넣어줄 v3의 row index
     * 
    **/
    static Vec2 vec3_to_vec2(const Vec3 &v3, int r0, int r1)
    {  
        assert(("r0, r1 should be <=2", r0 <= 2 && r1 <= 2));
        Vec2 v2 = Vec2::Zero();
        v2.row(0) = v3.row(r0);
        v2.row(1) = v3.row(r1);
        return v2;
    }

    /**
     * v2: 차원을  늘리고 싶은 vec2 
     * r0: input v2의 row(0)을 넣어줄 output vec3의 row의 idx (r0)
     * r1: input v2의 row(1)을 넣어줄 output vec3의 row의 idx (r1)
     * 
    **/
    static Vec3 vec2_to_vec3(const Vec2 &v2, int r0, int r1)
    {
        assert(("r0, r1 should be <=2", r0 <= 2 && r1 <= 2));
        Vec3 v3 = Vec3::Zero();
        v3.row(r0) = v2.row(0);
        v3.row(r1) = v2.row(1);
        return v3;
    }
    


    Tensor to_X_tensor(const PFNN_XY_FrameData &prev_data, const PFNN_XY_FrameData &curr_data)
    {
        // x_i = {t_p(i), t_d(i), t_h(i), t_g(i)-- 지금은 안씀, j_p(i-1), j_v(i-1)}

        //* Variables Setting
        int windowSize = curr_data.local_trj_positions.size();
        int jnt_size = curr_data.jnts_local_positions.size();

        //** Trajectory Position of Current Frame
        std::vector<Tensor> tensor_lists;
        for (int i = 0; i < windowSize; ++i)
        {
            Vec2 xz_pos = vec3_to_vec2(curr_data.local_trj_positions.at(i), 0, 2);
            Tensor pos_i = alt::vec_to_tensor(xz_pos);

            tensor_lists.push_back(pos_i);
        }

        //** Trajectory Direction of Current Frame
        for (int i = 0; i < windowSize; ++i)
        {
            Vec2 xz_dir = vec3_to_vec2(curr_data.local_trj_directions.at(i), 0, 2);
            Tensor dir_i = alt::vec_to_tensor(xz_dir);
            tensor_lists.push_back(dir_i);
        }

        //** Trajectory Heights of Current Frame
        for (int i = 0; i < windowSize; ++i)
        {
            Vec3 h = curr_data.local_trj_heights.at(i);
            Tensor h_i = alt::vec_to_tensor(h);
            tensor_lists.push_back(h_i);
        }

        //** Local Joints Position of the Previous Frame
        for (int i = 0; i < jnt_size; ++i)
        {
            Vec3 jnt_pos = prev_data.jnts_local_positions.at(i);
            Tensor pos_i = alt::vec_to_tensor(jnt_pos);
            tensor_lists.push_back(pos_i);
        }

        //** Local Joints Velocity of the Previous Frame
        for (int i = 0; i < jnt_size; ++i)
        {
            Vec3 jnt_vel = prev_data.jnts_local_velocity.at(i);
            Tensor vel_i = alt::vec_to_tensor(jnt_vel);
            tensor_lists.push_back(vel_i);
        }

        Tensor long_X_tensor = torch::cat(tensor_lists, 0); // [3 x noj]
        return long_X_tensor;
    };

    Tensor to_Y_tensor(const PFNN_XY_FrameData &curr_data, const PFNN_XY_FrameData &latt_data)
    {
        // y_i = {t_p(i+1), t_d(i+1), j_p(i), j_v(i), j_a(i), 
        //        r_x_dot(i), r_z_dot(i), r_ang_dot(i), dPhase, contact}
        
        //* Variables Setting
        int windowSize = curr_data.local_trj_positions.size();
        int jnt_size = curr_data.jnts_local_positions.size();
        std::vector<Tensor> tensor_lists;
        
        //** Trajectory Position of Latter Frame
        for (int i = 0; i < windowSize; ++i)
        {
            Vec2 xz_pos = vec3_to_vec2(latt_data.local_trj_positions.at(i), 0, 2);
            Tensor pos_i = alt::vec_to_tensor(xz_pos);

            tensor_lists.push_back(pos_i);
        }
        //** Trajectory Direction of Latter Frame
        for (int i = 0; i < windowSize; ++i)
        {
            Vec2 xz_dir = vec3_to_vec2(latt_data.local_trj_directions.at(i), 0, 2);
            Tensor dir_i = alt::vec_to_tensor(xz_dir);

            tensor_lists.push_back(dir_i);
        }

        //** Local Joints Position of the Current Frame        
        for (int i = 0; i < jnt_size; ++i)
        {
            Vec3 jnt_pos = curr_data.jnts_local_positions.at(i);
            Tensor pos_i = alt::vec_to_tensor(jnt_pos);
            tensor_lists.push_back(pos_i);
        }
        
        //** Local Joints Velocity of the Current Frame
        for (int i = 0; i < jnt_size; ++i)
        {
            Vec3 jnt_vel = curr_data.jnts_local_velocity.at(i);
            Tensor vel_i = alt::vec_to_tensor(jnt_vel);
            tensor_lists.push_back(vel_i);
        }

        //** Local Joints Angles of the Current Frame
        for (int i = 0; i < jnt_size; ++i)
        {
            Vec3 jnt_aaxis = curr_data.jnts_local_aaxis.at(i);
            Tensor aaxis_i = alt::vec_to_tensor(jnt_aaxis);
            tensor_lists.push_back(aaxis_i);
        }

        //** Local root Velocity of the Current Frame
        {
            Vec2 root_vel = vec3_to_vec2(curr_data.root_local_velocity, 0, 2);
            Tensor vel = alt::vec_to_tensor(root_vel);
            tensor_lists.push_back(vel);
        }

        //** Local root Angular Velocity of the Current Frame
        {
            Tensor ang_vel = torch::tensor({curr_data.root_angular_velocity});
            tensor_lists.push_back(ang_vel);
        }

        //** Foot Contact Label of Current Frame
        {
            Tensor contact = torch::tensor({curr_data.l_contact, curr_data.r_contact}, {torch::kBool});
            tensor_lists.push_back(contact);
        }

        //** Change in Phase of Current Frame
        {
            Tensor d_phase_ = torch::tensor({curr_data.dPhase});
            tensor_lists.push_back(d_phase_);
        }
        Tensor long_Y_tensor = torch::cat(tensor_lists, 0); // [3 x noj]

        return long_Y_tensor;
    };

    PFNN_Y_Data parse_Y_tensor_to_y_data(agl::spModel model,
                                      const float prev_phase,
                                      const Mat4 prev_rootTrf,
                                      const Tensor &Y_tensor,
                                      const int windowSize,
                                      const std::vector<std::string> joints_names)
    {
        const int jnt_size = joints_names.size();

        int size = 4 * windowSize + 9 * jnt_size + 6;
        assert(("size of Y_tensor and size of total split dimension have to be the same", 
                size == Y_tensor.size(0)));

        //** variable settings
        PFNN_Y_Data y_info_local;
        // local
        std::vector<Vec3> trj_pos_vector;
        std::vector<Vec3> trj_dir_vector;
        std::vector<Vec3> jnt_pos_vector;
        std::vector<Vec3> jnt_vel_vector;
        std::vector<Vec3> jnt_ang_vector;

        // Pose
        agl::Pose curr_pose;

        Mat4 curr_rootTrf = Mat4::Identity();
        const float timeStep = 1.0f / 60.0f;


        //** Step0) split long Y_tensor to short tensor
        // splits into {trj_pos, trj_dir, 
        //              jnt_pos, jnt_vel, jnt_ang, 
        //              root vel, root_angl_vel, 
        //              contact, dPhase}
        std::vector<Tensor> tensor_lists;
        {
            IntArrayRef split_size = {2 * windowSize, 2 * windowSize,
                                      3 * jnt_size, 3 * jnt_size, 3 * jnt_size,
                                      2, 1, 2, 1};

            for (auto t : torch::split_with_sizes(Y_tensor, split_size))
            {
                tensor_lists.push_back(t);
            }
        }

        //** Step1) Split Tensor into informative vectors
        {
            //** split trj_pos
            for (auto trj_pos : torch::chunk(tensor_lists.at(0), windowSize))
            {
                Vec2 xzPos_ = alt::tensor_to_vec(trj_pos);
                Vec3 xyzPos = vec2_to_vec3(xzPos_, 0, 2);
                trj_pos_vector.push_back(xyzPos);
            }

            //** split trj_dir
            for (auto trj_dir : torch::chunk(tensor_lists.at(1), windowSize))
            {
                Vec2 xzDir_ = alt::tensor_to_vec(trj_dir);
                Vec3 xyzDir = vec2_to_vec3(xzDir_, 0, 2);
                trj_dir_vector.push_back(xyzDir);
            }

            //** split jnt_pos
            for (auto jnt_pos : torch::chunk(tensor_lists.at(2), jnt_size))
            {
                Vec3 jntPos_ = alt::tensor_to_vec(jnt_pos);
                jnt_pos_vector.push_back(jntPos_);
            }

            //** split jnt_vel
            for (auto jnt_vel : torch::chunk(tensor_lists.at(3), jnt_size))
            {
                Vec3 jntVel_ = alt::tensor_to_vec(jnt_vel);
                jnt_vel_vector.push_back(jntVel_);
            }

            //** split jnt_ang
            for (auto jnt_ang : torch::chunk(tensor_lists.at(4), jnt_size))
            {
                Vec3 jntAng_ = alt::tensor_to_vec(jnt_ang);
                jnt_ang_vector.push_back(jntAng_);
            }

            y_info_local.latter_trj_lpos = trj_pos_vector;
            y_info_local.latter_trj_ldir = trj_dir_vector;
            y_info_local.current_jnt_lpos = jnt_pos_vector;
            y_info_local.current_jnt_lvel = jnt_vel_vector;
            y_info_local.current_jnt_laaxis = jnt_ang_vector;
        }

        {
            //** root_local_velocity
            {
                Vec2 r_xz_vel = alt::tensor_to_vec(tensor_lists.at(5));
                Vec3 r_xyz_vel = vec2_to_vec3(r_xz_vel, 0, 2);
                y_info_local.root_lvel = r_xyz_vel;
            }

            //** root_angular_velocity
            {
                y_info_local.root_ang_vel = (tensor_lists.at(6)).item<float>();
            }

            //** foot contact label
            {
                y_info_local.l_contact = std::round((tensor_lists.at(7))[0].item<float>());
                y_info_local.r_contact = std::round((tensor_lists.at(7))[1].item<float>());
            }

            //** current phase
            {
                float phase = prev_phase + (tensor_lists.at(8)).item<float>();
                // std::cout << "dPhase is: " << (tensor_lists.at(8)).item<float>() << std::endl;
                //! 만약 phase가 0 ~ 2pi 사이가 아니라면?
                // if phase is 2 * M_PI
                if((phase - (2 * M_PI)) < 1e-7)
                {
                    y_info_local.phase = phase;

                }
                // except when phase is (2 * M_PI), use modulus function
                else
                {
                    y_info_local.phase = fmod(phase, 2 * M_PI);
                }
            }
        }

        //** Step2) Calculate current rootTrf from previous rootTrf
        {
            //** Calculate current root transformation
            // get global position of rootTransform
            Vec3 root_gl_pos = prev_rootTrf.col(3).head(3) +
                               (prev_rootTrf.block<3, 3>(0, 0) * y_info_local.root_lvel) * timeStep;


            // get global rotation of rootTransform
            float dTheta = y_info_local.root_ang_vel * timeStep;
            Mat rq = AAxis(dTheta, Vec3::UnitY()).toRotationMatrix();
            Mat3 root_gl_rot = prev_rootTrf.block<3, 3>(0, 0) * rq;

            // update current rootTrf
            curr_rootTrf.block<3, 3>(0, 0) = root_gl_rot;
            curr_rootTrf.col(3).head(3) = root_gl_pos;
            y_info_local.world_root_trf = curr_rootTrf;
        }

        //** Step3) Calculate latter_trj_lh
        //! needs to be changed when the height is not zero
        y_info_local.latter_trj_lh.resize(windowSize, Vec3::Zero());

        return y_info_local;
    }
   
    Tensor to_X_tensor_from_Y(const PFNN_Y_Data& y_data)
    {
        // x_i = {t_p(i), t_d(i), t_h(i), t_g(i)-- 지금은 안씀, j_p(i-1), j_v(i-1)}

        //* Variables Setting
        int windowSize = y_data.latter_trj_lpos.size();
        int jnt_size = y_data.current_jnt_lpos.size();

        //** Trajectory Position of Current Frame
        std::vector<Tensor> tensor_lists;
        for (int i = 0; i < windowSize; ++i)
        {
            Vec2 xz_pos = vec3_to_vec2(y_data.latter_trj_lpos.at(i), 0, 2);
            Tensor pos_i = alt::vec_to_tensor(xz_pos);
            tensor_lists.push_back(pos_i);
        }

        //** Trajectory Direction of Current Frame
        for (int i = 0; i < windowSize; ++i)
        {
            Vec2 xz_dir = vec3_to_vec2(y_data.latter_trj_ldir.at(i), 0, 2);
            Tensor dir_i = alt::vec_to_tensor(xz_dir);
            tensor_lists.push_back(dir_i);
        }

        //** Trajectory Heights of Current Frame
        for (int i = 0; i < windowSize; ++i)
        {
            Vec3 h = y_data.latter_trj_lh.at(i);
            Tensor h_i = alt::vec_to_tensor(h);
            tensor_lists.push_back(h_i);
        }

        //** Local Joints Position of the Previous Frame
        for (int i = 0; i < jnt_size; ++i)
        {
            Vec3 jnt_pos = y_data.current_jnt_lpos.at(i);
            Tensor pos_i = alt::vec_to_tensor(jnt_pos);
            tensor_lists.push_back(pos_i);
        }

        //** Local Joints Velocity of the Previous Frame
        for (int i = 0; i < jnt_size; ++i)
        {
            Vec3 jnt_vel = y_data.current_jnt_lvel.at(i);
            Tensor vel_i = alt::vec_to_tensor(jnt_vel);
            tensor_lists.push_back(vel_i);
        }

        Tensor long_X_tensor = torch::cat(tensor_lists, 0); // [3 x noj]
        return long_X_tensor;
    };


    Tensor to_X_tensor_from_YTensor(const Tensor& y_tensor, int windowSize, int jnt_size)
    {
        // x_i = {t_p(i), t_d(i), t_h(i), t_g(i)-- 지금은 안씀, j_p(i-1), j_v(i-1)}
        torch::Device device = torch::kCPU;
        
        // make sure that input and output are on the same device
        if(y_tensor.is_cuda())
        {
            device = torch::kCUDA;
        }

        // indices
        Tensor t_range = torch::range(0, 4 * windowSize - 1, at::device(device).dtype(torch::kInt64));
        Tensor j_range = torch::range(4 * windowSize, 4 * windowSize + 6 * jnt_size - 1, 
                                      at::device(device).dtype(torch::kInt64));
        
        Tensor traj_info = torch::index_select(y_tensor, 0, t_range);
        Tensor jnt_info = torch::index_select(y_tensor, 0, j_range);

        

        //! this has to be changed when the heights are not zeros
        Tensor traj_height_info = torch::zeros(3 * windowSize, at::device(device));

        Tensor XTensor = torch::cat({traj_info, traj_height_info, jnt_info});

        return XTensor;
    };


    agl::Pose reconstruct_pose(agl::spModel model,
                               const PFNN_Y_Data& y_data,
                               const std::vector<std::string> joints_names)
    {
        //** variable settings
        // global
        std::vector<Mat3> jnt_global_rot;
        std::vector <Vec3> jnt_global_pos;
        const int jnt_size = y_data.current_jnt_laaxis.size();

        // Pose
        agl::Pose curr_pose;

        Mat4 curr_rootTrf = y_data.world_root_trf;

        //** Step1) Get Global Rotation of joints from local Rotation of joints
        for (int i = 0; i < jnt_size; ++i)
        {
            Vec3 local_aaxis = y_data.current_jnt_laaxis.at(i);
            float angle = local_aaxis.norm();
            Vec3 axis = local_aaxis / angle;
            Mat3 jnt_l_rot = AAxis(angle, axis).toRotationMatrix();
            Mat3 jnt_g_rot = curr_rootTrf.block<3, 3>(0, 0) * jnt_l_rot;
            jnt_global_rot.push_back(jnt_g_rot);
        }


        //** Step2) Get Global position of joints
        for (int i = 0; i < jnt_size; ++i)
        {
            Vec3 l_pos = y_data.current_jnt_lpos.at(i);
            Vec4 l_pos_homo = Vec4::Ones();
            l_pos_homo.block<3, 1>(0, 0) = l_pos;

            Vec3 g_pos = (curr_rootTrf * l_pos_homo).head(3);
            jnt_global_pos.push_back(g_pos);
        }

        //** Step3) stack Pose information
        {
            //** set local rotation of model's joints
            for (int i = 0; i < joints_names.size(); ++i)
            {
                std::string jnt = joints_names.at(i);

                if(model->joint(jnt)->parent() != nullptr)
                {
                    agl::spJoint parent = model->joint(jnt)->parent();
                    Quat parent_inv = (parent->world_rot()).inverse();
                    Quat global_rot = Quat(jnt_global_rot.at(i));

                    Quat local_rot = parent_inv * global_rot;
                    model->joint(jnt)->set_local_rot(local_rot);
                }
                else
                {
                    Quat global_rot = Quat(jnt_global_rot.at(i));
                    model->joint(jnt)->set_local_rot(global_rot);
                }
            }
            
            model->root()->update_world_trf_children(); // update global trfs

            
            for (auto jnt : model->joints())
            {
                curr_pose.local_rotations.push_back(jnt->local_rot());
            }
            curr_pose.root_position = jnt_global_pos.at(0);
        }
        
        return curr_pose;
    };

    static Vec3 blend_traj_direction(const Mat4 rootTrf,
                              Vec3 orig_trj_dir,
                              Vec3 target_dir,
                              const float dir_scale)
    {
        assert(("0 <= dir_scale <= 1 should be satisfied", dir_scale <= 1 || dir_scale >= 0));

        // when there is no user control
        if (target_dir.norm() < 1e-05)
        {
            target_dir = orig_trj_dir;
        }

        orig_trj_dir.normalize();
        target_dir.normalize();
        // std::cout << "orig_trj_dir: " << orig_trj_dir << std::endl;

        // get rotation from orig_dir and target_dir
        Quat orig_trj_dir_q = Quat(AAxis(atan2(orig_trj_dir.block<1,1>(0,0).value(),
                                               orig_trj_dir.block<1,1>(2,0).value()), 
                                               Vec3::UnitY()));

        Quat target_dir_q = Quat(AAxis(atan2(target_dir.block<1,1>(0,0).value(), 
                                             target_dir.block<1,1>(2,0).value()), 
                                             Vec3::UnitY()));

        orig_trj_dir_q.normalize();
        target_dir_q.normalize();

        // slerp original and target direction
        Quat mixed_q = orig_trj_dir_q.slerp(dir_scale, target_dir_q);

        // if(abs(dir_scale - 1) < 0.000001)
        // {
        //     std::cout << "***** blend_traj_direction *****" << std::endl;
        //     std::cout << "dir_scale: " << dir_scale << std::endl;

        //     // mixed_q.normalize();
        //     std::cout << "target_dir_q:\n" << target_dir_q.toRotationMatrix() << std::endl;
        //     std::cout << "orig_trj_dir_q:\n" << orig_trj_dir_q.toRotationMatrix() << std::endl;
        //     std::cout << "mixed_q:\n" << mixed_q.toRotationMatrix() << std::endl;
        //     std::cout << "\n";
        // }

        return mixed_q * Vec3::UnitZ();

    }

    static Vec3 blend_traj_position(const Mat4& rootTrf,
                             const Vec3& prev_orig_trj_pos,
                             const Vec3& curr_orig_trj_pos,
                             const Vec3& prev_blended_trj_pos,
                             const Vec3& target_vel,
                             const float pos_scale)
    {
        assert(("0 <= pos_scale <= 1 should be satisfied", pos_scale <= 1 || pos_scale >= 0));

        Vec3 vel = target_vel;
        vel = 0.2f * vel;


        // std::cout << "vel: " << vel.block<1,1>(0,0) <<"  " << vel.block<1,1>(1,0) <<"  " << vel.block<1,1>(2,0) << std::endl;

        // when there is no user control set velocity to original velocity
        if(target_vel.norm() < 1e-05)
        {
            vel = curr_orig_trj_pos - prev_orig_trj_pos;
        }

        Vec3 blended_vel = (1 - pos_scale) * (curr_orig_trj_pos - prev_orig_trj_pos) 
                            + pos_scale * vel;

        Vec3 curr_blended_trj_pos = prev_blended_trj_pos + blended_vel;

        return curr_blended_trj_pos;
    }

    vVec3 blend_traj_poss(const Mat4& rootTrf,
                          const vVec3& trj_orig_pos,
                          const float pos_bias,
                          const Vec3& target_vel,
                          const int windowSize)
    {
        assert(("windowSize and length of trj_orig_pos should be same", trj_orig_pos.size() == windowSize));

        vVec3 blended_trj_pos = trj_orig_pos;
        for (int i = (windowSize / 2) + 1; i < windowSize; ++i)
        {
            // float scale = (float) i / (windowSize / 2 - 1) - 1;
            float scale = (i - windowSize / 2) / (float)(windowSize / 2 - 1);
            float pos_scale = powf(scale, pos_bias);

            blended_trj_pos.at(i) = blend_traj_position(rootTrf, trj_orig_pos.at(i - 1), trj_orig_pos.at(i), 
                                                        blended_trj_pos.at(i - 1), target_vel, pos_scale);
        }
        return blended_trj_pos;
    }

    vVec3 blend_traj_dirs(const Mat4 rootTrf,
                          const vVec3 trj_orig_dir,
                          const float dir_bias,
                          const Vec3 target_dir,
                          const int windowSize)
    {
        assert(("windowSize and length of trj_orig_pos should be same", trj_orig_dir.size() == windowSize));

        vVec3 blended_trj_dir = trj_orig_dir;
        for (int i = windowSize / 2 + 1; i < windowSize; ++i)
        {
            float scale = (float) i / (windowSize / 2) - 1;
            float dir_scale = powf(scale, dir_bias);

            blended_trj_dir.at(i) = blend_traj_direction(rootTrf, trj_orig_dir.at(i), target_dir, dir_scale);
        }

        return blended_trj_dir;
    }


    Tensor jntInfo_fromRecon(agl::spModel model,
                                const agl::Pose& pose,
                                const Mat4& rootTrf,
                                const std::vector<std::string>& joint_names,
                                const std::vector<Vec3>& prev_jnt_glob_pos)
    {
        // variables setting
        const int jnt_num = joint_names.size();

        std::vector<Tensor> jnts_local_positionT;
        std::vector<Tensor> jnts_local_velocityT;

        std::vector<Vec3> jnts_local_pos;

        jnts_local_positionT.resize(jnt_num);
        jnts_local_pos.resize(jnt_num);
        jnts_local_velocityT.resize(jnt_num);


        // // set pose
        // model->set_pose(pose);
        // model->update_mesh();

        //** get joint local position
        for (int j = 0; j < jnt_num; ++j)
        {
            //get name of each joint
            std::string jnt = joint_names.at(j);

            // get jnt_local_pos
            Vec4 jnt_world_pos_homo = Vec4::Ones(4);
            jnt_world_pos_homo.block<3, 1>(0, 0) = model->joint(jnt)->world_pos();
            Vec3 jnt_local_pos = (rootTrf.inverse() * jnt_world_pos_homo).block<3, 1>(0, 0);
            jnts_local_pos.at(j) = jnt_local_pos;

            // change Vec3 to Tensor data            
            Tensor jnt_pos_ = alt::vec_to_tensor(jnt_local_pos);
            jnts_local_positionT.at(j) = jnt_pos_;
        }

        //*get joint local velocity
        for (int j = 0; j < jnt_num; ++j)
        {
            // get previous jnt_local_pos
            Vec4 prev_jnt_world_pos_homo = Vec4::Ones(4);
            prev_jnt_world_pos_homo.block<3, 1>(0, 0) = prev_jnt_glob_pos.at(j);
            Vec3 prev_jnt_local_pos = (rootTrf.inverse() * prev_jnt_world_pos_homo).block<3, 1>(0, 0);
            
            //get velocity info
            Vec3 jnt_local_vel = (jnts_local_pos.at(j) - prev_jnt_local_pos) / (1.0f / 60.0f);

            Tensor jnt_vel = alt::vec_to_tensor(jnt_local_vel);
            jnts_local_velocityT.at(j) = jnt_vel;
        }

        Tensor pos_t = torch::cat(jnts_local_positionT, 0);
        Tensor vel_t = torch::cat(jnts_local_velocityT, 0);

        Tensor result = torch::cat({pos_t, vel_t});

        return result;
    }

    void prev_trajInfo_fromRecon(Tensor& trj_prev_lposT, 
                            Tensor& trj_prev_ldirT,
                            vVec3& trj_prev_lposV, 
                            vVec3& trj_prev_ldirV, 
                            const vVec3& trj_glb_prev_pos, 
                            const vVec3& trj_glb_prev_dir,
                            const float windowSize,
                            const Mat4& rt,
                            const std::vector<int> sample_timing)
    {

        std::vector <Tensor> trj_prev_lpos;
        std::vector <Tensor> trj_prev_ldir;

        trj_prev_lpos.resize(windowSize/2);
        trj_prev_ldir.resize(windowSize/2);

        trj_prev_lposV.resize(windowSize/2);
        trj_prev_ldirV.resize(windowSize/2);

        for (int i = 0; i < windowSize / 2; ++i)
        {
            Vec3 trj_prev_glb_pos = trj_glb_prev_pos.at(sample_timing.at(i) + 60);
            Vec3 trj_prev_glb_dir = trj_glb_prev_dir.at(sample_timing.at(i) + 60);

            Vec4 trj_prev_glb_pos_homo = Vec4::Ones();
            trj_prev_glb_pos_homo.block<3, 1>(0, 0) = trj_prev_glb_pos;
            
            Vec4 trj_prev_glb_dir_homo = Vec4::Zero();
            trj_prev_glb_dir_homo.block<3, 1>(0, 0) = trj_prev_glb_dir;

            Vec3 trj_prev_l_pos = (rt.inverse() * trj_prev_glb_pos_homo).block<3, 1>(0, 0);
            Vec3 trj_prev_l_dir = (rt.inverse() * trj_prev_glb_dir_homo).block<3, 1>(0, 0);

            Vec2 trj_prev_l_pos_ = vec3_to_vec2(trj_prev_l_pos, 0, 2);
            Vec2 trj_prev_l_dir_ = vec3_to_vec2(trj_prev_l_dir, 0, 2);

            Tensor trj_l_pos = alt::vec_to_tensor(trj_prev_l_pos_);
            Tensor trj_l_dir = alt::vec_to_tensor(trj_prev_l_dir_);

            trj_prev_lpos.at(i) = trj_l_pos;
            trj_prev_ldir.at(i) = trj_l_dir;
            
            trj_prev_lposV.at(i) = vec2_to_vec3(trj_prev_l_pos_, 0, 2);
            trj_prev_ldirV.at(i) = vec2_to_vec3(trj_prev_l_dir_, 0, 2);
        }

        trj_prev_lposT = torch::cat(trj_prev_lpos, 0);
        trj_prev_ldirT = torch::cat(trj_prev_ldir, 0);
    }

    void latter_trajInfo_fromRecon(Tensor &trj_lat_lposT,
                                   Tensor &trj_lat_ldirT,
                                   const vVec3 &trj_lpos,
                                   const vVec3 &trj_ldir,
                                   const float windowSize)
    {
        std::vector<Tensor> trj_lat_lpos;
        std::vector<Tensor> trj_lat_ldir;

        trj_lat_lpos.resize(windowSize / 2);
        trj_lat_ldir.resize(windowSize / 2);

        for (int i = windowSize / 2; i < windowSize; ++i)
        {
            Vec3 trj_lp = trj_lpos.at(i);
            Vec3 trj_ld = trj_ldir.at(i);

            Vec2 trj_p = vec3_to_vec2(trj_lp, 0, 2);
            Vec2 trj_d = vec3_to_vec2(trj_ld, 0, 2);

            trj_lat_lpos.at(i - windowSize / 2) = alt::vec_to_tensor(trj_p);
            trj_lat_ldir.at(i - windowSize / 2) = alt::vec_to_tensor(trj_d);
        }

        trj_lat_lposT = torch::cat(trj_lat_lpos, 0);
        trj_lat_ldirT = torch::cat(trj_lat_ldir, 0);
    };
}