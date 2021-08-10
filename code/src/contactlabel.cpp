#include "aPFNN/contactlabel.h"
#include <iostream>

namespace a::pfnn {

static void changeContactInfo(const int pose_num, 
                              std::vector<bool> &foot_contact_label, 
                              const int currentFrame, 
                              const int missingValue)
{
    for (int i = 0; i < missingValue; ++i)
    {
        foot_contact_label.at((currentFrame + i) % pose_num) = true;
    }
}

static bool checkContactofFormerFrames(const int pose_num, 
                                       const int currentFrame, 
                                       const std::vector<bool>& foot_contact_label, 
                                       const int lower_bound, 
                                       bool wasFormerStanding)
{

    for (int j = 1; j < lower_bound + 1; ++j)
    {
        //int formerFrame = (currentFrame - j < 0) ? (pose_num + (currentFrame - j) % pose_num) : ((currentFrame - j) % pose_num);
        int formerFrame = (currentFrame - j < 0) ? 0 : ((currentFrame - j) % pose_num);
        if (foot_contact_label.at(formerFrame))
        {
            wasFormerStanding = true;
            continue;
        }
        else
        {
            wasFormerStanding = false;
            break;
        }
    }
    return wasFormerStanding;
}

static bool check_isMissingCountCorrect(const int pose_num, 
                                     const int currentFrame, 
                                     const std::vector<bool> &foot_contact_label, 
                                     const int missing_num, 
                                     bool isMissingCountCorrect)
{
    for (int j = 0; j < missing_num; ++j)
    {
        int frame = ((currentFrame + j) > (pose_num -1)) ? (pose_num - 1) : (currentFrame + j);

        if (foot_contact_label.at(frame))
        {
            isMissingCountCorrect = false;
            break;
        }
    }
    return isMissingCountCorrect;
}

static bool checkContactofLatterFrames(const int pose_num, 
                                       const int currentFrame, 
                                       const std::vector<bool> &foot_contact_label, 
                                       const int upper_bound, 
                                       const int missing_num, 
                                       bool isLatterStanding)
{

    for (int j = 1; j < (upper_bound + 1); ++j)
    {
        int latterFrame = (currentFrame + j > pose_num - 1) ? (pose_num - 1) : ((currentFrame + missing_num - 1) + j);
        
        // latter frame이 만약에 주어진 pose의 개수보다 넘어가면
        if (latterFrame > pose_num -1)
        {
            isLatterStanding = false;
            break;
        }

        if (foot_contact_label.at(latterFrame))
        {
            isLatterStanding = true;
            continue;
        }
        else
        {
            isLatterStanding = false;
            break;
        }
    }
    return isLatterStanding;
}

float approximate_standard_foot_height(const agl::spJoint FootToeBase,
                                       const agl::spJoint FootToeEnd,
                                       const agl::spJoint Foot)
{
    Vec3 pos_FootToeBase = FootToeBase->world_pos();
    Vec3 pos_FootToeEnd = FootToeEnd->world_pos();
    Vec3 pos_Foot = Foot->world_pos();

    Vec3 dir_endToBase = (pos_FootToeBase - pos_FootToeEnd).normalized();
    Vec3 dir_endToFoot = (pos_Foot - pos_FootToeEnd).normalized();

    float angle = agl::safe_acos(dir_endToBase.dot(dir_endToFoot));
    float hypotenuse = (pos_Foot - pos_FootToeEnd).norm();
    float foot_height = hypotenuse * sin(angle) + pos_FootToeEnd(1,0);

    return foot_height;
}


void label_contact_info(std::vector<bool> &left_foot_contact_label,
                        std::vector<bool> &right_foot_contact_label,
                        agl::spModel model,
                        const agl::Motion &motion,
                        agl::spJoint leftFoot,
                        agl::spJoint rightFoot,
                        const float foot_height_standard,
                        const float height_epsilon,
                        const float speed_epsilon,
                        const float timeStep)
{
    float leftFootHeight = foot_height_standard;
    float rightFootHeight = foot_height_standard;

    // pose information setting
    int pose_num = motion.poses.size();
    left_foot_contact_label.resize(pose_num);
    right_foot_contact_label.resize(pose_num);

    // feet position of former frame and current frame
    Vec3 leftFoot_formerPos;
    Vec3 leftFoot_currentPos;
    Vec3 rightFoot_formerPos;
    Vec3 rightFoot_currentPos;

    // check contact
    for (int i = 0; i < pose_num; ++i)
    {
        // former Position of feet
        leftFoot_formerPos = leftFoot->world_pos();
        rightFoot_formerPos = rightFoot->world_pos();

        // set model
        model->set_pose(motion.poses.at(i));
        model->root()->update_world_trf_children(); // update global trfs

        // current Position of feet
        leftFoot_currentPos = leftFoot->world_pos();
        rightFoot_currentPos = rightFoot->world_pos();

        // detect left_foot contact
        {
            // initialize contact as false
            left_foot_contact_label.at(i) = false;

            // check proximity - whether adjacent joint(toe) is sufficiently close to the ground
            float leftFoot_yPos = leftFoot_currentPos(1, 0);
            bool isLeftOnGround = (std::abs(leftFoot_yPos - leftFootHeight) < height_epsilon) ? true : false;

            // check relative velocity - whether its velocity is below some threshold
            Vec3 leftFoot_velocity = (leftFoot_currentPos - leftFoot_formerPos) / timeStep;
            float leftFoot_speed = leftFoot_velocity.norm();

            bool isLeftSpeedZero = (leftFoot_speed < speed_epsilon) ? true : false;

            // set contact info
            if (isLeftOnGround && isLeftSpeedZero)
            {
                left_foot_contact_label.at(i) = true;
            }
        }
        // detect right_foot contact
        {
            // initialize contact as false
            right_foot_contact_label.at(i) = false;

            // check proximity - whether adjacent joint(toe) is sufficiently close to the ground
            float rightFoot_yPos = rightFoot_currentPos(1, 0);
            bool isRightOnGround = (std::abs(rightFoot_yPos - rightFootHeight) < height_epsilon) ? true : false;

            // check relative velocity - whether its velocity is below some threshold
            Vec3 rigtFoot_velocity = (rightFoot_currentPos - rightFoot_formerPos) / timeStep;
            float rightFoot_speed = rigtFoot_velocity.norm();

            bool isRightSpeedZero = (rightFoot_speed < speed_epsilon) ? true : false;

            // set contact info
            if (isRightOnGround && isRightSpeedZero)
            {
                right_foot_contact_label.at(i) = true;
            }
        }

    }
}

void filter_contact_info(std::vector<bool> &left_foot_contact_label,
                         std::vector<bool> &right_foot_contact_label,
                         const int former_window,
                         const int latter_window,
                         const int maximum_missing)
{
    // pose information setting
    assert(("# of left and right foot label has to be same", left_foot_contact_label.size() == left_foot_contact_label.size()));
    int pose_num = left_foot_contact_label.size();

    // check whether number of missing value is correct
    bool left_isMissingCountCorrect = true;
    bool right_isMissingCountCorrect = true;

    bool left_wasFormerStanding = false;
    bool left_isLatterStanding = false;

    bool right_wasFormerStanding = false;
    bool right_isLatterStanding = false;

    // find missing values
    for (int k = maximum_missing; k > 0; --k)
    {
        for (int i = 0; i < pose_num; ++i)
        {
            //initialize values
            left_isMissingCountCorrect = true;
            right_isMissingCountCorrect = true;

            left_wasFormerStanding = false;
            left_isLatterStanding = false;

            right_wasFormerStanding = false;
            right_isLatterStanding = false;

            // 비교할 frame들이 pose_num의 사이즈를 넘어가는지 확인하기
            int lower_bound = former_window;
            int upper_bound = latter_window;
            //clamp(i, lower_bound, upper_bound, pose_num, k);

            if (!left_foot_contact_label.at(i))
            {
                // check whether left foot was contacting at former frames (former_window만큼의 frame)
                left_wasFormerStanding = checkContactofFormerFrames(pose_num, i, left_foot_contact_label, lower_bound, left_wasFormerStanding);

                // check whether left foot is not contacting for # of missingValue at later frames
                left_isMissingCountCorrect = check_isMissingCountCorrect(pose_num, i, left_foot_contact_label, k, left_isMissingCountCorrect);

                if (left_isMissingCountCorrect)
                {
                    // check whether left foot is contacting at latter frames (latter_window만큼의 frame)
                    left_isLatterStanding = checkContactofLatterFrames(pose_num, i, left_foot_contact_label, upper_bound, k, left_isLatterStanding);
                }

                // filter_contact_info
                if (left_wasFormerStanding && left_isLatterStanding)
                {
                    changeContactInfo(pose_num, left_foot_contact_label, i, k);
                }
            }

            if (!right_foot_contact_label.at(i))
            {
                // check whether right foot was contacting at former frames (former_window만큼의 frame)
                right_wasFormerStanding = checkContactofFormerFrames(pose_num, i, right_foot_contact_label, lower_bound, right_wasFormerStanding);

                // check whether right foot is not contacting for # of missingValue at later frames
                right_isMissingCountCorrect = check_isMissingCountCorrect(pose_num, i, right_foot_contact_label, k, right_isMissingCountCorrect);

                if (right_isMissingCountCorrect)
                {
                    // check whether right foot is contacting at latter frames (latter_window만큼의 frame)
                    right_isLatterStanding = checkContactofLatterFrames(pose_num, i, right_foot_contact_label, upper_bound, k, right_isLatterStanding);
                }

                // filter_contact_info
                if (right_wasFormerStanding && right_isLatterStanding)
                {
                    changeContactInfo(pose_num, right_foot_contact_label, i, k);
                }
            }

        }
    }
}

}