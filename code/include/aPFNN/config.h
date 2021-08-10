#pragma once
#include <aOpenGL.h>

namespace a::pfnn
{
    struct ContactPhaseInfo
    {
        std::vector<bool> left_contact;
        std::vector<bool> right_contact;
        std::vector<float> phase;
    };

    struct RootInfo
    {
        std::vector<Mat4> root_world_trf;
    };

    struct PFNN_XY_FrameData
    {
        Mat4 world_root_trf;
        std::vector<Vec3> local_trj_positions;  // size: window size
        std::vector<Vec3> local_trj_directions; // size: window size
        std::vector<Vec3> local_trj_heights;    // size: window size
        
        std::vector<Vec3> jnts_local_positions; // size: noj
        std::vector<Vec3> jnts_local_aaxis;     // size: noj
        std::vector<Vec3> jnts_local_velocity;  // size: noj

        Vec3 root_local_velocity;               // unit: 1/s
        float root_angular_velocity;            // unit: rad/s
        bool l_contact;
        bool r_contact;
        float dPhase;                           // unit: rad
    };

    struct PFNN_Y_Data
    {
        Mat4 world_root_trf;
        std::vector<Vec3> latter_trj_lpos;        // size: window size
        std::vector<Vec3> latter_trj_ldir;        // size: window size
        std::vector<Vec3> latter_trj_lh;          // size: window size
        
        std::vector<Vec3> current_jnt_lpos;       // size: noj
        std::vector<Vec3> current_jnt_laaxis;     // size: noj
        std::vector<Vec3> current_jnt_lvel;       // size: noj

        Vec3 root_lvel;               // unit: 1/s
        float root_ang_vel;            // unit: rad/s
        bool l_contact;
        bool r_contact;
        float phase;                           // unit: rad
    };

    static const std::vector<std::string> kmodel_jnt_names = {
        "Hips",
        "LeftUpLeg", "LeftLeg", "LeftFoot", 
        // "LeftToeBase", "LeftToe_End",
        "RightUpLeg", "RightLeg", "RightFoot", 
        // "RightToeBase", "RightToe_End",
        "Spine", "Spine1", "Spine2", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",

        //** children of LeftHand
        // "LeftHandThumb1","LeftHandThumb2","LeftHandThumb3","LeftHandThumb4",
        // "LeftHandIndex1", "LeftHandIndex2","LeftHandIndex3","LeftHandIndex4",
        // "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4",
        // "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4",
        // "LeftHandPinky1", "LeftHandPinky2","LeftHandPinky3", "LeftHandPinky4",

        "RightShoulder", "RightArm", "RightForeArm", "RightHand",

        //** children of RightHand
        // "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumb4",
        // "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4",
        // "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4",
        // "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4",
        // "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4",

        "Neck", "Head", 
        // "HeadTop_End"
        };

    static const std::vector<std::string> anubis_joints_names =
        {
            "Hips",
            //"LeftUpLeg", "LeftLeg", 
            "LeftFoot", 
            // "LeftToeBase", "LeftFootIndex1", "LeftFootIndex2", "LeftFootMiddle1", "LeftFootMiddle2", "LeftFootRing1", "LeftFootRing2", "LeftFootPinky1", "LeftFootPinky2", "LeftFootExtraFinger1", "LeftFootExtraFinger2", "LeftFootJewelry",
            // "RightUpLeg", "RightLeg", 
            "RightFoot", 
            // "RightToeBase", "RightFootIndex1", "RightFootIndex2", "RightFootMiddle1", "RightFootMiddle2", "RightFootRing1", "RightFootRing2", "RightFootPinky1", "RightFootPinky2", "RightFootExtraFinger1", "RightFootExtraFinger2", "RightFootJewelry",
            // "Spine", "Spine1", "Spine2",
            "LeftShoulder", 
            // "LeftArm", "LeftForeArm", "LeftWrist", 
            "LeftHand", 
            // "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
            // "Neck", "Head", "L_eye", "R_eye", "L_ear1", "L_ear2", "L_ear3", "R_ear1", "R_ear2", "R_ear3",
            // "Jaw_Pivot", "Jaw", "Chin", 
            // "Tongue05", "Tongue04", "Tongue03", "Tongue02", "Tongue01",
            "RightShoulder",
            // "RightArm", "RightForeArm", "RightWrist", 
            "RightHand",
            // "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
            // "Tasset1", "Tasset2", "Tasset3", "Tasset4"
        };

    static const std::vector<std::string> anubis_motion_joint_names =
        {
            "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            //"LeftFootIndex1", "LeftFootIndex2", "LeftFootMiddle1", "LeftFootMiddle2", "LeftFootRing1", "LeftFootRing2", "LeftFootPinky1", "LeftFootPinky2", "LeftFootExtraFinger1", "LeftFootExtraFinger2", "LeftFootJewelry",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
            //"RightFootIndex1", "RightFootIndex2", "RightFootMiddle1", "RightFootMiddle2", "RightFootRing1", "RightFootRing2", "RightFootPinky1", "RightFootPinky2", "RightFootExtraFinger1", "RightFootExtraFinger2", "RightFootJewelry",
            "Spine", "Spine1", "Spine2",
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftWrist", "LeftHand",
            //"LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
            "Neck", "Head",
            //"L_eye", "R_eye", "L_ear1", "L_ear2", "L_ear3", "R_ear1", "R_ear2", "R_ear3", "Jaw_Pivot", "Jaw", "Chin", "Tongue05", "Tongue04", "Tongue03", "Tongue02", "Tongue01",
            "RightShoulder", "RightArm", "RightForeArm", "RightWrist", "RightHand"
            //"RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
            //"Tasset1", "Tasset2", "Tasset3", "Tasset4"
    };

    static const std::vector<int>
        sample_timings = { -60, -50, -40, -30, -20, -10,
                          0, 10, 20, 30, 40, 50};

    struct hparam
    {
        int inputSize;
        int outputSize;
        int batchSize;
        int numberOfEpochs;
        int checkpoint;
        float dprob;

    }; 
}