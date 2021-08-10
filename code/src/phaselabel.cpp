#include "aPFNN/phaselabel.h"
#include <iostream>

namespace a::pfnn{

    static bool check_wasTrulyStanding(const std::vector<bool>& left_foot_contact_label, 
                                const std::vector<bool>& right_foot_contact_label,
                                const int current_idx,
                                const int later_idx)
    {
        for(int i = current_idx; i <= later_idx; ++i)
        {
            // 하나의 발이라도 지면과 떨어졌다면 standing하고 있지 않은 상태
            if(!left_foot_contact_label.at(i) || !right_foot_contact_label.at(i))
            {
                return false;
            }
        }
        return true;
    }
    
    static void interpolate_phase(std::vector<float>& phase, 
                           const int start_idx, 
                           const int end_idx, 
                           const float start_phase, 
                           const float later_phase)
    {
        int isZerotoPi;
        // phase 변화가 [ 0 ~ M_PI ]일 때
        if(abs((later_phase - start_phase) - M_PI) < 0.01f)
        {
            isZerotoPi = 1;
        }
        // phase 변화가 [ M_PI ~ 0 (2 * M_PI) ]일때 
        else if(abs((start_phase - later_phase) - M_PI) < 0.01f)
        {
            isZerotoPi = -1;
        }
        // ! phase 변화가 [ 0 ~ 0 ]  or [ M_PI ~ M_PI ] 일때 - 특이점임. 이후 임의로 interpolation해주어야함.
        else
        {
            isZerotoPi = 0;
        }

        int stride =0;

        switch(isZerotoPi)
        {
        // if the phase has to be interpolated between zero to M_PI
        case (1):
            stride = end_idx - start_idx;
            for(int i = 1; i < stride; i++)
            {
                phase.at(start_idx + i) = (M_PI / stride) * i;
            }
            break;
            // if the phase has to be interpolated between M_PI to 2_PI
        case (-1):
            stride = end_idx - start_idx;
            for (int i = 1; i < stride; i++)
            {
                phase.at(start_idx + i) = M_PI + (M_PI / stride) * i;
            }
            break;

        case (0):
            break;

        default:
            break;
        }
    }
    
    static void interpolate_standingPhase(std::vector<float>& phase, 
                                   const int start_idx, 
                                   const int end_idx, 
                                   const float start_phase, 
                                   const float later_phase, 
                                   const int min_frame_of_contact)
    {
        const float epsilon = 0.1f;
        // 0 ~ 0 or M_PI ~ M_PI interpolation ; 정수배 cycle씩 돌아가는 것.
        if(abs(later_phase - start_phase) < epsilon)
        {
            const int cycle = round((double)(end_idx - start_idx + 1) / min_frame_of_contact);
            const float stride = (float)(end_idx - start_idx) / (2 * cycle);

            // make checkpoint frame
            for (int i = 1; i <= (2 * cycle); i++)
            {
                int former_idx = round(start_idx + stride * (i - 1));
                int next_idx = round(start_idx + stride * i);

                phase.at(next_idx) = (abs(M_PI - phase.at(former_idx)) < epsilon) ? 0 : M_PI;

                interpolate_phase(phase, former_idx, next_idx, phase.at(former_idx), phase.at(next_idx));
            }
        }
        // 0 ~ M_PI or  M_PI ~ 0 interpolation ; 0.5 *(2n + 1) cycle씩 돌아가는 것. (n은 자연수)
        else
        {
            const int cycle = ((end_idx - start_idx + 1) / min_frame_of_contact);
            const float stride = (float)(end_idx - start_idx) / (2 * cycle + 1);

            // make checkpoint frame
            for (int i = 1; i <= (2 * cycle + 1); i++)
            {
                int isZerotoPi = 0;

                int former_idx = round(start_idx + stride * (i - 1));
                int next_idx = round(start_idx + stride * i);

                phase.at(next_idx) = (abs(M_PI - phase.at(former_idx)) < epsilon) ? 0.0f : M_PI;
                interpolate_phase(phase, former_idx, next_idx, phase.at(former_idx), phase.at(next_idx));
            }
        }
    }
 
    static void interpolate_emptyValues(std::vector<float>& phase,                            
                            const int start_idx, 
                            const int end_idx,
                            const int min_frame_of_contact)
    {
        int step = end_idx - start_idx;
        //assert((abs(phase.at(start_idx) -1) < epsilon) || (abs(phase.at(end_idx) -1) < epsilon));
        const float epsilon = 0.01f;
        const float rest_value = (float) M_PI / 180;
        const float stride = (2 * M_PI) / min_frame_of_contact;
        if (abs(phase.at(start_idx) - rest_value) < epsilon)
        {
            for (int i = 1; i < step + 1; ++i)
            {
                float candidate_phase = (phase.at(end_idx) - stride * i);
                float new_phase = (candidate_phase > 0) ? candidate_phase : (2 * M_PI - std::fmod(candidate_phase, 2 * M_PI));
                phase.at(end_idx - i) = new_phase;
            }
        }
        else if(abs(phase.at(end_idx) - rest_value) < epsilon)
        {
            for (int i = 1; i < step + 1; ++i)
            {
                phase.at(start_idx + i) = std::fmod(phase.at(start_idx) + stride * i , 2 * M_PI);
            }
        }
        // there is no need to interpolate (no missing values)
        else
        {
            return;
        }
    }                            
                       
 
    void label_phase_info(const std::vector<bool> &left_foot_contact_label,
                          const std::vector<bool> &right_foot_contact_label,
                          std::vector<float> &phase,
                          const int min_frame_of_contact)
    {
        assert(left_foot_contact_label.size() == left_foot_contact_label.size());

        int pose_num = left_foot_contact_label.size();
        phase.resize(pose_num);
        
        // <index of frame, phase> that has 0 or M_PI
        std::map<int, float> checkpoint_phase;

        // checkpoint_phase(0 or M_PI)가 아닌 다른 phase들의 initialization value
        const float non_value = M_PI / 180.0f;

        std::map<int, float>::iterator iter;
        const float epsilon = 0.01f;
        
        
        //**  make checkpoint frame - check phase of [0 or M_PI]
        for (int i = 0; i < pose_num; ++i)
        {
            int formerFrame = ((i - 1) > 0) ? (i - 1) : 0;
            int latterFrame = ((i + 1) > pose_num - 1) ? (pose_num - 1) : (i + 1);

            // when left foot comes in contact with the ground, then phase is M_PI
            if (!left_foot_contact_label.at(formerFrame) && left_foot_contact_label.at(i))
            {
                phase.at(i) = M_PI;
                checkpoint_phase.insert(std::make_pair(i, phase.at(i)));
                continue;
            }

            // when right foot comes in contact with the ground, then phase is 0
            else if (!right_foot_contact_label.at(formerFrame) && right_foot_contact_label.at(i))
            {
                phase.at(i) = 0.0f;
                checkpoint_phase.insert(std::make_pair(i, phase.at(i)));
                continue;
            }
            else if (left_foot_contact_label.at(i) && right_foot_contact_label.at(i))
            {
                if (!left_foot_contact_label.at(latterFrame))
                {
                    phase.at(i) = 0.0f;
                    checkpoint_phase.insert(std::make_pair(i, phase.at(i)));
                    continue;
                }
                else if (!right_foot_contact_label.at(latterFrame))
                {
                    phase.at(i) = M_PI;
                    checkpoint_phase.insert(std::make_pair(i, phase.at(i)));
                    continue;
                }
            }
            phase.at(i) = non_value;
        }
        
        
        //** smooth interpolation of phases between two checkpoint_phase 
        for (iter = checkpoint_phase.begin(); iter != --checkpoint_phase.end(); iter++)
        {
            iter++;
            int later_idx = iter->first;
            float later_phase = iter->second;

            iter--;
            int current_idx = iter->first;
            float current_phase = iter->second;

            // 연속되는 두 checkpoint frame의 phase가 same이면, contact time 확인
            if (abs(later_phase - current_phase) < epsilon)
            {
                // count # of frame
                // 정해진 frame보다 적게 standing하고 있다면, 그 checkpoint는 지우기 (standing하고 있지 않다고 판단)
                if ((later_idx - current_idx + 1) < min_frame_of_contact)
                {
                    
                    bool wasTrulyStanding = check_wasTrulyStanding(left_foot_contact_label, right_foot_contact_label, current_idx, later_idx);

                    if (wasTrulyStanding)
                    {
                        phase.at(later_idx) = non_value;
                        checkpoint_phase.erase(later_idx);
                        // 그 이후의 checkpoint_frame과의 interpolation하기
                        iter++;
                        later_idx = iter->first;
                        later_phase = iter->second;

                        interpolate_phase(phase, current_idx, later_idx, current_phase, later_phase);
                        iter--;
                    }
                    else
                    {
                        // 특이점들 - 임의로 interpolation해준다.
                        interpolate_standingPhase(phase, current_idx, later_idx, current_phase, later_phase, later_idx - current_idx + 1);
                    }
                }
                // 정해진 frame보다 많이 standing하고 있다면, standing하고 있다고 판단하기
                else
                {
                    // interpolate standing cycles (정수배 cycle로 iterate)
                    interpolate_standingPhase(phase, current_idx, later_idx, current_phase, later_phase, min_frame_of_contact);
                }
            }
            // 연속되는 두 checkpoint frame이 0 ~ M_PI 또는 M_PI ~ 0 까지라면, 그 사이의 frame들 동안 모든 발이 contact하고 있었는지 확인하기
            else
            {
                bool wasTrulyStanding = check_wasTrulyStanding(left_foot_contact_label, right_foot_contact_label, current_idx, later_idx);

                // interpolate standing cycles (0.5*n cycle로 iterate - 이때 n은 3 이상의 홀수)
                if (wasTrulyStanding)
                {
                    interpolate_standingPhase(phase, current_idx, later_idx, current_phase, later_phase, min_frame_of_contact);
                }
                // interpolate smoothly
                else
                {
                    interpolate_phase(phase, current_idx, later_idx, current_phase, later_phase);
                }
            }
        }

        //** interpolation of rest phase that has not been interpolated yet
        //(첫번째 checkpoint frame 나오기 이전의 frames의 smooth interpolation
        interpolate_emptyValues(phase, 0, checkpoint_phase.begin()->first, min_frame_of_contact);
        // 마지막 checkpoint frame 이후에 나오는 frames의 smooth interpolation
        interpolate_emptyValues(phase, (--checkpoint_phase.end())->first, pose_num - 1, min_frame_of_contact);

        //! 특이점들에 대해서는 (non-value를 연속적으로 가지는(label이 되지 않은 frame들)은 임의로 phase가 1 cycle을 돌도록 함.)
        for (iter = checkpoint_phase.begin(); iter != --checkpoint_phase.end(); iter++)
        {
            iter++;
            int later_idx = iter->first;
            float later_phase = iter->second;

            iter--;
            int current_idx = iter->first;
            float current_phase = iter->second;
            if (abs(later_phase - current_phase) < epsilon)
            {

                for (int j = current_idx + 1; j < later_idx; j++)
                {
                    if (abs(phase.at(j) - non_value) > epsilon)
                    {
                        break;
                    }
                    interpolate_standingPhase(phase, current_idx, later_idx, current_phase, later_phase, later_idx - current_idx + 1);
                }
            }
        }
    }




}