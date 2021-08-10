#include "aPFNN/utils.h"

namespace a::pfnn{

    std::vector<Tensor> normalize_tensor(const Tensor &feature)
    {
        assert(feature.dim() == 1);

        IntArrayRef dim({0});
        auto std_mean = torch::std_mean(feature, dim, true, false);
        Tensor stdev = std::get<0>(std_mean);
        Tensor mean = std::get<1>(std_mean);

        Tensor normalized_features = feature - mean;
        normalized_features.div_(stdev);
        return {normalized_features, mean, stdev};
    }

    Tensor normalize_tensors(const Tensor &features, const int dimen)
    {
        assert(features.dim() == 2);
        assert(dimen == 1 || dimen == 0);
        std::vector<Tensor> t_lists;
        t_lists.resize(features.size(dimen));

        IntArrayRef dim({0});
        

        for (int i = 0; i < features.size(dimen); ++i)
        {
            Tensor t = torch::squeeze(torch::narrow(features, dimen, i, 1));
            

            auto std_mean = torch::std_mean(t, dim, true, false);
            Tensor stdev = std::get<0>(std_mean);

            Tensor mean = std::get<1>(std_mean);

            Tensor normalized_t = t - mean;

            normalized_t.div_(stdev);
            //! if all variables are same, stdev becomes 0, so do not divide with it 
            //! the variable is (heights of trajectories)
            if ((stdev.item<float>() - 0.0f) < 1e-10)
            {
                normalized_t = t - mean;
            }
            t_lists.at(i) = normalized_t;
        }
        Tensor output = torch::vstack(t_lists);
        if (output.sizes() != features.sizes())
        {
            output = torch::transpose(output, 0, 1);
        }

        return output;
    }


    Tensor normalize_tensor_with_stdev(const Tensor &feature, const Tensor &mean_stdev, const Tensor &range)
    {
        assert(feature.dim() == 1);
        assert(feature.size(0) == range.size(0));
        
        torch::Device device = torch::kCPU;
        if(feature.is_cuda() || mean_stdev.is_cuda())
        {
            device = torch::kCUDA;
        }
        Tensor index_selected = torch::index_select(mean_stdev.to(device), 0, range.to(device)).to(device);

        Tensor mean = torch::squeeze(torch::narrow(index_selected, 1, 0, 1));
        Tensor stdev = torch::squeeze(torch::narrow(index_selected, 1, 1, 1));

        Tensor normalized_t = feature.to(device) - mean;
        normalized_t.div_(stdev);

        return normalized_t;
    }


    std::vector<std::vector<Tensor>> normalize_tensors_seperate(const Tensor &features, const int dimen)
    {

        assert(features.dim() == 2);
        assert(dimen == 1 || dimen == 0);

        std::vector<std::vector<Tensor>> t_lists;
        t_lists.resize(features.size(dimen));

        for (int i = 0; i < features.size(dimen); ++i)
        {
            IntArrayRef dim({0});

            Tensor t = torch::squeeze(torch::narrow(features, dimen, i, 1));

            auto std_mean = torch::std_mean(t, dim, true, false);

            Tensor stdev = std::get<0>(std_mean);
            Tensor mean = std::get<1>(std_mean);

            Tensor normalized_t = t - mean;
            normalized_t.div_(stdev);

            // //! if all variables are same, stdev becomes 0, so do not divide with it 
            // //! the variable is (heights of trajectories)
            if ((stdev.item<float>() - 0.0f) < 1e-10)
            {
                normalized_t = t - mean;
            }

            //! dim({0}) 을 for문 밖에 선언하면 이 연산으로 인해 바뀜.. why?! ㅠㅠ
            t_lists.at(i) = {normalized_t, mean, stdev};

        }

        return t_lists;
    }

    Tensor scale_tensor(const Tensor &features, const Tensor &range_of_idx, const float scale)
    {
        assert(features.dim() == 1 || features.dim() == 2);

        torch::Device device = torch::kCPU;
        if(features.is_cuda())
        {
            device = torch::kCUDA;
        }

        Tensor mask;
        Tensor scaled_feature = torch::zeros_like(features).to(device);


        if(features.dim() == 1)
        {
            mask = torch::ones_like(features).to(device);
            for(int i = 0; i < range_of_idx.size(0); ++i)
            {
                mask[range_of_idx[i]] = scale;
            }
            scaled_feature = features * mask;
        }

        if(features.dim() == 2)
        {
            mask = torch::ones(features.size(1)).to(device);
            for (int i = 0; i < range_of_idx.size(0); ++i)
            {
                mask[range_of_idx[i]] = scale;
            }
            
            for (int i = 0; i < features.size(0); ++i)
            {
                scaled_feature[0, i] = features[0, i] * mask;
            }
        }

        return scaled_feature                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ;
    }

    Tensor denormalize_tensor(const Tensor &features, Tensor& mean_stdev)
    {
        assert(features.dim() == 1);

        torch::Device device = torch::kCPU;
        
        // make sure that input and output are on the same device
        if(features.is_cuda())
        {
            device = torch::kCUDA;
            mean_stdev = mean_stdev.to(device);
        }

        Tensor mean = torch::squeeze(torch::narrow(mean_stdev, 1, 0, 1));
        Tensor stdev = torch::squeeze(torch::narrow(mean_stdev, 1, 1, 1));
        // Tensor mean = mean_stdev.at(0);
        // Tensor stdev = mean_stdev.at(1);

        Tensor t = features;
        // t = t * stdev + mean;
        t = t.mul(stdev);
        t = t.add(mean);
        return t;
    }


    Tensor to_int_type_tensor(const Tensor &feature, const Tensor&range_of_idx)
    {
        assert(feature.dim() == 1);

        torch::Device device = torch::kCPU;
        if (feature.is_cuda())
        {
            device = torch::kCUDA;
        }

        return feature;
    };

}