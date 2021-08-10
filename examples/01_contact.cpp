#include <aPfnn.h>
#include <iostream>

class MyApp : public agl::App
{
public:

    void draw_local_coord(Mat4 trf, float scale = 2.0f)
    {
        //color of x-axis; R
        //color of y-axis; G
        //color of z-axis; B
        Mat4 xOffset = Mat4::Identity();
        xOffset(0, 3) = scale * 0.05f;

        Mat4 yOffset = Mat4::Identity();
        yOffset(1, 3) = scale * 0.05f;
        
        Mat4 zOffset = Mat4::Identity();
        zOffset(2, 3) = scale * 0.05f;
        
        agl::Render::cube()
            ->transform(trf * xOffset)
            ->color(1, 0, 0)
            //->scale(scale * 0.1f, 0.01f, 0.01f)
            ->scale(scale * 0.1f, 0.04f, 0.04f)
            ->debug(true)
            ->draw();
        
        agl::Render::cube()
            ->transform(trf * yOffset)
            ->color(0, 1, 0)
            //->scale(0.01f, scale * 0.1f, 0.01f)
            ->scale(0.04f, scale * 0.1f, 0.04f)
            ->debug(true)
            ->draw();

        agl::Render::cube()
            ->transform(trf * zOffset)
            ->color(0, 0, 1)
            ->scale(0.04f, 0.04f, scale * 0.1f)
            ->debug(true)
            ->draw();
    }

    agl::spModel model;
    std::vector<agl::Motion> motions;
    apfnn::ContactPhaseInfo contact_info;
    apfnn::RootInfo root_info;
    std::vector<float> phase;


    int update_freq = 1;

    int frame = 12000 * update_freq;
    // ubi_sprint1_subject2: 7200 15454
    // ubi_sprint1_subject4: 6354
    int pidx;

    void start() override
    {
        const char* model_path  = "../data/fbx/kmodel/model/kmodel.fbx";
        const char* motion_path = "../data/fbx/kmodel/motion/ubi_sprint1_subject4.fbx";

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

        // todo
        float foot_height = apfnn::approximate_standard_foot_height(LeftToeBase, LeftToe_End, leftFoot);
        apfnn::label_contact_info(contact_info.left_contact, contact_info.right_contact, model, motions.at(0), 
                                leftFoot, rightFoot, foot_height, 0.06f, 1.13f, (1.0f / 60.0f));
        apfnn::filter_contact_info(contact_info.left_contact, contact_info.right_contact, 1, 1, 4);
        
        apfnn::label_phase_info(contact_info.left_contact, contact_info.right_contact, contact_info.phase, 30);

        apfnn::get_root_info(root_info.root_world_trf, model, motions.at(0), LeftShoulder, RightShoulder, Hips);

    }

    void update() override
    {
        const auto& motion = motions.at(0);
        pidx = frame / update_freq;
        const auto& pose = motion.poses.at(pidx);
        model->set_pose(pose);
        model->update_mesh();
        frame = (frame + 1) % (int)(motion.poses.size() * update_freq);
        
        // camera fix
        {
            Vec3 r_axis = Vec3::UnitY();
            float sangle = 3 * M_PI / 2;

            Mat3 m3 = AAxis(sangle, r_axis).toRotationMatrix();
            Mat4 wtrf = model->root()->world_trf();
            wtrf.block<3, 3>(0, 0) = m3;
            wtrf.block<3, 3>(0, 0) = Mat3::Identity();
            
            wtrf(1, 3) = 0.5f;

            Vec4 cam_local_pos(0.0f, 2.0f, 4.0f, 1.0f);
            Vec4 cam_world_pos = wtrf * cam_local_pos;
            
            glm::vec3 cam_focus_pos(agl::to_glm((Vec4)wtrf.col(3)));
            
            glm::mat4 gtrf = a::gl::to_glm(wtrf);
            camera().set_focus(cam_focus_pos);
            glm::vec3 cam_pos_glm(agl::to_glm(cam_world_pos));
            camera().set_position(cam_pos_glm);


        }
    }
    

    void render() override
    {

        agl::Render::plane()
            ->scale(500.0f)
            ->color(0.15f, 0.15f, 0.15f)
            ->floor_grid(true)
            ->draw();

        agl::Render::model(model)
            ->draw();
        
        //bool is_left_contact = info.left_contact.at(frame);
        
        bool is_right_contact = contact_info.right_contact.at(pidx);
        bool is_left_contact = contact_info.left_contact.at(pidx);

        Vec3 rightFoot = model->joint("RightFoot")->world_pos();
        Vec3 leftFoot = model->joint("LeftFoot")->world_pos();
        Vec3 Hips = model->joint("Hips")->world_pos();

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
                ->color(1, 0, 0)
                ->debug(true)
                ->draw();
        }

        // float angle = 1* M_PI/2;
        // Mat3 rotation = root_info.root_world_trf.at(pidx).block<3, 3>(0, 0);
        // Vec3 ph = rotation * Vec3{0.3f * -cos(contact_info.phase.at(pidx)), 0, -0.3f * sin(contact_info.phase.at(pidx))};
        
        // float weight = abs(sin(0.5f * contact_info.phase.at(pidx)));
        // Vec3 clr = weight * Vec3(0, 1, 0) + (1.0f - weight) * Vec3(1, 0, 0);
        // agl::Render::sphere()
        //     ->position(Hips + ph)
        //     ->scale(0.2f)
        //     ->color(clr)
        //     ->debug(true)
        //     ->draw();
    }
        

    void render_xray() override
    {
        agl::Render::skeleton(model)
            ->color(0.9, 0.9, 0)
            ->draw();

        // draw_local_coord(model->joint("Hips")->world_trf());
        // draw_local_coord(root_info.root_world_trf.at(pidx));
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
    //app.set_window_size(1.5f * 1920, 1.5f * 1080);
    agl::AppManager::start(&app);
    return 0;
}