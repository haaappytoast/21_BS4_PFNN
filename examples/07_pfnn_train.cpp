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

    apfnn::hparam param{204, 234, 32, 40, 50, 0.7};

    std::vector<int> sample_timing;
    const int windowSize = 12;

    int update_freq = 1;

    int frame = 0* update_freq;
    int pidx;

    bool restore = false;
    bool useVal = true;
    void start() override
    {
        //** STEP1 variables setting
        const char *model_path = "../data/fbx/kmodel/model/kmodel.fbx";
        const char *motion_path = "../data/fbx/kmodel/motion/simple3.fbx";

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

        std::cout << "joint_names.size(): " << joint_names.size() << std::endl;     // 20

        framesData.resize(motions.at(0).poses.size());
        //** STEP2 data setting (contact labeling, root info extracting, root info extracting)
        float foot_height = apfnn::approximate_standard_foot_height(LeftToeBase, LeftToe_End, leftFoot);

        apfnn::label_contact_info(info.left_contact, info.right_contact, model, motions.at(0),
                                  leftFoot, rightFoot, foot_height, 0.06f, 1.13f, (1.0f / 60.0f));

        apfnn::filter_contact_info(info.left_contact, info.right_contact, 1, 1, 4);
        
        apfnn::label_phase_info(info.left_contact, info.right_contact, phase, 30);
        
        apfnn::get_root_info(root_info.root_world_trf, model, motions.at(0), LeftShoulder, RightShoulder, Hips);

        //** STEP3 get data for each frame
        int min_frame = -sample_timing.at(0);                                   // 60
        int max_frame = (framesData.size() - sample_timing.at(windowSize - 1)); // simple3.fbx에선 271

        // for (int i = min_frame; i < max_frame; ++i)
        // {
        //     framesData.at(i) = apfnn::get_frame_data(i, model, motions.at(0).poses, 
        //                                            root_info.root_world_trf, phase, 
        //                                            joint_names, sample_timing, 
        //                                            info.left_contact.at(i), info.right_contact.at(i));
        //     if(i % 1000 == 0 || i == max_frame - 1)
        //     {
        //         std::cout << i << " th XYframeData setting done" << std::endl;
        //     }
        // }

        std::cout << "************* XYframeData Setting Done *************\n" << std::endl;

        std::vector<Tensor> X_tensor_lists, Y_tensor_lists;
        Tensor XTensor, YTensor, phaseTensor;
        
        Tensor XTensor_normed, YTensor_normed, mean_stdev_t;

        Tensor val_range, train_range;
        
        Tensor x_train, y_train, x_val, y_val, ph_train, ph_val;

        //** STEP4 get X,Y tensor data from frameData
        if (0)
        {
            // 1. PhaseTensor Data
            std::vector<float> filtered_phase;
            for (int i = min_frame + 1; i < max_frame - 1; ++i)
            {   
                filtered_phase.push_back(phase.at(i));
            }

            phaseTensor = torch::tensor(filtered_phase);
            torch::save(phaseTensor, "./ptfiles/tensors/phaseTensor.pt");
            std::cout << "************* phaseTensor saved *************\n" << std::endl;

            // 2. X,Y Tensor
            for (int i = min_frame + 1; i < max_frame - 1; ++i)
            {
                Tensor x_i = apfnn::to_X_tensor(framesData.at(i - 1), framesData.at(i));
                Tensor y_i = apfnn::to_Y_tensor(framesData.at(i), framesData.at(i + 1));
                X_tensor_lists.push_back(x_i);
                Y_tensor_lists.push_back(y_i);
                if (i % 1000 == 0 || i == max_frame - 1)
                {
                    std::cout << i << " th Tensor Data setting done" << std::endl;
                }
            }
            //** Change it into Large Tensor
            XTensor = torch::vstack(X_tensor_lists);
            YTensor = torch::vstack(Y_tensor_lists);

            // std::cout << torch::index_select(XTensor[38], 0, torch::range(0, 24, torch::kInt64)) << std::endl;
            // std::cout << framesData.at(99).local_trj_positions << std::endl;
            // exit(0);
            std::cout << "XTensor.sizes(): " << XTensor.sizes() << std::endl; // [14157, 234]
            std::cout << "YTensor.sizes(): " << YTensor.sizes() << std::endl; // [14157, 279]
            std::cout << "\n************* Data Setting Done *************\n"
                      << std::endl;

            torch::save(XTensor, "./ptfiles/tensors/Xtensor.pt");
            torch::save(YTensor, "./ptfiles/tensors/Ytensor.pt");
            
            val_range = torch::range(101, 200, torch::kInt64);
            train_range = torch::cat({torch::range(0, 100, torch::kInt64),
                                             torch::range(201, XTensor.size(0) - 1, torch::kInt64)}, 0);

            torch::save(val_range, "./ptfiles/tensors/val_range.pt");
            torch::save(train_range, "./ptfiles/tensors/train_range.pt");

            std::cout << "XTensor, YTensor, validation_range, train_range saved" << std::endl;
        }

        else
        {
            torch::load(XTensor, "./ptfiles/tensors/Xtensor.pt");
            torch::load(YTensor, "./ptfiles/tensors/Ytensor.pt");
            torch::load(phaseTensor, "./ptfiles/tensors/phaseTensor.pt");

            torch::load(val_range, "./ptfiles/tensors/val_range.pt");
            torch::load(train_range, "./ptfiles/tensors/train_range.pt");

            std::cout << "XTensor.sizes(): " << XTensor.sizes() << std::endl; // [14157, 234]
            std::cout << "YTensor.sizes(): " << YTensor.sizes() << std::endl; // [14157, 279]
            std::cout << "phaseTensor.sizes(): " << phaseTensor.sizes() << std::endl; // [14157, 279]
            std::cout << "*************XTensor, YTensor, phaseTensor loaded*************\n\n" << std::endl;
        }

        // mean & stdev check
        // std::vector<std::vector<Tensor>> stdm_list = apfnn::normalize_tensors_seperate(YTensor, 1);
        // for (int i = 0; i < YTensor.size(1); ++i)
        // {
        //     Tensor p = torch::squeeze(torch::narrow(YTensor, 1, i, 1));
        //     std::cout << "1. mean: " << apfnn::normalize_tensor(p).at(1).item<float>() << std::endl;
        //     std::cout << "1. stdev: " << apfnn::normalize_tensor(p).at(2).item<float>() << std::endl;
            
        //     std::cout << "\n2. mean: " << stdm_list.at(i).at(1).item<float>()  << std::endl;
        //     std::cout << "2. stdev: " << stdm_list.at(i).at(2).item<float>() << std::endl;
        //     std::cout << " -----------------\n" << std::endl;
        // }

        // normalization test
        // YTensor_normed = apfnn::normalize_tensors(YTensor, 1);
        // for (int i = 0; i < YTensor_normed.size(1); ++i)
        // {
        //     Tensor p = torch::squeeze(torch::narrow(YTensor, 1, i, 1));
        //     std::cout << "1. Tensor summation: " << p.sum().item<float>() << std::endl;
        //     std::cout << "1. mean: " << apfnn::normalize_tensor(p).at(1).item<float>() << std::endl;
        //     std::cout << "1. stdev: " << apfnn::normalize_tensor(p).at(2).item<float>() << std::endl;
        //     float sum = 0.0f;
        //     std::cout << std::endl;

        //     for (int j = 0; j < YTensor.size(0); ++j)
        //     {
        //         sum += YTensor[j][i].item<float>();
        //     }
        //     std::cout << "2. summation: " << sum << std::endl;
        //     std::cout << "2. mean: " << sum / YTensor.size(0) << std::endl;
        //     std::cout << " -----------------\n" << std::endl;

        // }

        // exit(0);

        if(0)
        {
            XTensor_normed = apfnn::normalize_tensors(XTensor, 1);
            YTensor_normed = apfnn::normalize_tensors(YTensor, 1);
            torch::save(XTensor_normed, "./ptfiles/tensors/normed_Xtensor.pt");
            torch::save(YTensor_normed, "./ptfiles/tensors/normed_Ytensor.pt");


            // scale XTensor and YTensor for elements related to joints
            XTensor_normed = apfnn::scale_tensor(XTensor_normed, torch::range(7 * windowSize, XTensor.size(1) - 1, torch::kInt32), 0.1f);
            YTensor_normed = apfnn::scale_tensor(YTensor_normed, torch::range(4 * windowSize, YTensor.size(1) - 7, torch::kInt32), 0.1f);

            // get mean and stdev of YTensors for reconstruction
            std::vector<std::vector<Tensor>> mean_stdev_list = apfnn::normalize_tensors_seperate(YTensor, 1);
           
            std::vector<Tensor> mean_stdev_tensor;
            for (auto mean_stdev : mean_stdev_list)
            {
                Tensor meandev = torch::tensor({mean_stdev.at(1).item<float>(), mean_stdev.at(2).item<float>()}, torch::kFloat32);
                mean_stdev_tensor.push_back(meandev);
            }
            mean_stdev_t = torch::vstack(mean_stdev_tensor);
            // std::cout << "left dev:" << mean_stdev_t[231][0] << "left mean:" << mean_stdev_t[231][1] << std::endl;  
            // std::cout << "right dev:" << mean_stdev_t[232][0] << "right mean:" << mean_stdev_t[232][1] << std::endl;  
            // std::cout << "mean_stdev_t: " << mean_stdev_t[0, 0][1] << std::endl;

            torch::save(XTensor_normed, "./ptfiles/tensors/final_Xtensor.pt");
            torch::save(YTensor_normed, "./ptfiles/tensors/final_Ytensor.pt");
            torch::save(mean_stdev_t, "./ptfiles/tensors/mean_stdev.pt");

            std::cout << "\n\nXTensor.sizes(): " << XTensor.sizes() << std::endl; // [14157, 204]
            std::cout << "YTensor.sizes(): " << YTensor.sizes() << std::endl;     // [14157, 234]
            std::cout << "*************final_Xtensor, final_Ytensor saved*************\n\n"
                      << std::endl;
            std::cout << "mean_stdev_tensor saved\tsize: " << mean_stdev_t.sizes() << "\n\n"
                      << std::endl;

            // change X,YTensor to normed X,YTensors
            XTensor = XTensor_normed;
            YTensor = YTensor_normed;
        }

        else
        {
            torch::load(XTensor, "./ptfiles/tensors/final_Xtensor.pt");
            torch::load(YTensor, "./ptfiles/tensors/final_Ytensor.pt");
            torch::load(mean_stdev_t, "./ptfiles/tensors/mean_stdev.pt");
            std::cout << "final_XTensor.sizes(): " << XTensor.sizes() << std::endl; // [14157, 204]
            std::cout << "final_YTensor.sizes(): " << YTensor.sizes() << std::endl; // [14157, 234]
            std::cout << "mean_stdev_t.sizes(): " << mean_stdev_t.sizes() << std::endl; // [14157, 234]
            std::cout << "*************scaled XTensor, scaled YTensor loaded*************\n\n"
                      << std::endl;
        }

        if(useVal)
        {
            x_train = torch::index_select(XTensor, 0, train_range);
            x_val = torch::index_select(XTensor, 0, val_range);
            y_train = torch::index_select(YTensor, 0, train_range);
            y_val = torch::index_select(YTensor, 0, val_range);
            ph_train = torch::index_select(phaseTensor, 0, train_range);
            ph_val = torch::index_select(phaseTensor, 0, val_range);
            std::cout << " *************** Use Training Set and Validation Set ***************" << std::endl;
        }

        //** device setting
        torch::Device device = torch::kCPU;

        if (torch::cuda::is_available())
        {
            std::cout << "CUDA is available! Training on GPU." << std::endl;
            device = torch::kCUDA;

            XTensor = XTensor.to(device);
            YTensor = YTensor.to(device);
            mean_stdev_t = mean_stdev_t.to(device);
            phaseTensor = phaseTensor.to(device);

            x_train = x_train.to(device);
            x_val = x_val.to(device);
            y_train = y_train.to(device);
            y_val = y_val.to(device);
            ph_train = ph_train.to(device);
            ph_val = ph_val.to(device);
        }

        //** network setting
        apfnn::PFNN pfnn{nullptr};
        pfnn = apfnn::PFNN(param.inputSize, param.outputSize, true, param.dprob);    
        Adam pfnn_optimizer(pfnn->parameters(), AdamOptions().lr(1e-4));
        pfnn->to(device);

        int batches_per_epoch = (int) (x_train.size(0) / param.batchSize) + 1;

        int checkpointCount = 0;

    if (0)
    {
        torch::load(pfnn, "./ptfiles/pfnn_pts/pfnn-checkpoint.pt");
        torch::load(pfnn_optimizer, "./ptfiles/pfnn_pts/pfnn_optimizer-checkpoint.pt");
    }

    std::cout << "pfnn->is_train: " <<pfnn->is_train << std::endl;
    
    // since it is training network, make sure that the network is on training
    if(pfnn->is_train != true)
    {
        pfnn->is_train = true;
    }
    
    Tensor meanLoss = torch::zeros(1).to(device);

    for (int epoch = 0; epoch < param.numberOfEpochs; ++epoch)
    {            
        int batch_idx = 0;
        meanLoss = torch::zeros(1).to(device);
        Tensor random_idx = torch::randperm(x_train.size(0)).to(device);
        for(int j = 0; j < batches_per_epoch; ++j)
        {
            pfnn->zero_grad();
            //** generate random index for batchSize and get input data from XTensor
            int start_idx = 32 * j;
            int end_idx = (j < (batches_per_epoch -1)) ? 32 * (j + 1): x_train.size(0);
            assert(("end_idx should be less than x_train.size(0)", end_idx < x_train.size(0) + 1));
            //! shouldn't use randint because the numbers can be overlapped 
            // Tensor indices = torch::randint(x_train.size(0), {param.batchSize}, at::device(device).dtype(torch::kInt64));
            Tensor index_sel = torch::arange(start_idx, end_idx).to(device);
            Tensor indices = torch::index_select(random_idx, 0, index_sel).to(device).toType(torch::kLong);
            Tensor inputs = torch::index_select(x_train, 0, indices);


            //** phase info
            Tensor phases = torch::index_select(ph_train, 0, indices);

            //** real output from YTensor
            Tensor gtruth = torch::index_select(y_train, 0, indices);
            gtruth = gtruth.to(device);

            Tensor output = pfnn->forward(inputs, phases);
            output.squeeze_(1);
            
            output = apfnn::scale_tensor(output, torch::range(4 * windowSize, YTensor.size(1) - 7, torch::kInt32), 10.0f);
            gtruth = apfnn::scale_tensor(gtruth, torch::range(4 * windowSize, YTensor.size(1) - 7, torch::kInt32), 10.0f);

            Tensor loss = torch::mse_loss(output, gtruth);
            meanLoss += loss;


            //** BackProp
            loss.backward();
            pfnn_optimizer.step();


            //** checkpoint
            if (batch_idx % param.checkpoint == 0)
            {
                // Checkpoint the model and optimizer state.
                torch::save(pfnn, "./ptfiles/pfnn_pts/pfnn-checkpoint.pt");
                torch::save(pfnn_optimizer, "./ptfiles/pfnn_pts/pfnn_optimizer-checkpoint.pt");

                std::cout << "\n\n-> checkpoint " << ++checkpointCount << std::endl;
                std::cout << std::endl;
            }
            std::printf(
                "\r[%2d/%2d][%3d/%3d] loss: %.4f",
                epoch + 1,
                param.numberOfEpochs,
                ++batch_idx,
                batches_per_epoch,
                meanLoss.item<float>() / j);
        }
        
        //** change learning rate
        // if((epoch + 1) % 10 == 0)
        // {
        //     for (auto &group : pfnn_optimizer.param_groups())
        //     {
        //         if (group.has_options())
        //         {
        //             auto &options = static_cast<torch::optim::AdamOptions &>(group.options());
        //             options.lr(options.lr() * (1.0 - 0.3));
        //             std::cout <<"\noptions.lr(): " << options.lr() << std::endl;
        //         }
            
        //     }   

        if(useVal)
        {

            int bt_per_epoch = (int) (x_val.size(0) / param.batchSize) + 1;
            Tensor meanLoss = torch::zeros(1).to(device);
            Tensor random_idx = torch::randperm(x_val.size(0)).to(device);

            for(int j = 0; j < bt_per_epoch; ++j)
            {
                int start_idx = 32 * j;
                int end_idx = (j < (bt_per_epoch -1)) ? 32 * (j + 1): x_val.size(0);
                assert(("end_idx should be less than x_train.size(0)", end_idx < x_val.size(0) + 1));
                Tensor index_sel = torch::arange(start_idx, end_idx).to(device);
                Tensor val_indices = torch::index_select(random_idx, 0, index_sel).to(device).toType(torch::kLong);
                
                Tensor val_inputs = torch::index_select(x_val, 0, val_indices).to(device);
                Tensor val_phases = torch::index_select(ph_val, 0, val_indices).to(device);
                Tensor val_gtruth = torch::index_select(y_val, 0, val_indices).to(device);

                Tensor val_output = pfnn->forward(val_inputs, val_phases);

                val_output.squeeze_(1);

                Tensor loss = torch::mse_loss(val_output, val_gtruth);
                meanLoss += loss;
            }
            std::cout << "\n\n###### validation set loss: " << meanLoss.item<float>() / bt_per_epoch << " ######" << std::endl;
        }
    }
    
}

    void update() override
    {
        const auto& motion = motions.at(0);
        pidx = frame / update_freq;
        const auto& pose = motion.poses.at(pidx);
        model->set_pose(pose);
        //frame_data = framesData.at(pidx);
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
    }
        

    void render_xray() override
    {
        agl::Render::skeleton(model)
            ->color(0.9, 0.9, 0)
            ->draw();
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
        
    }
};

int main(int argc, char* argv[])
{
    MyApp app;
    agl::AppManager::start(&app);
    return 0;
}