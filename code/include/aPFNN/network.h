#pragma once
#include <aLibTorch.h>

namespace a::pfnn {

//class fcnet;

#define L1_UNIT 256
#define L2_UNIT 256
class fcNetImpl : public nnModule
{
public:
    int input_size, output_size;
    bool is_train;
    float dprob;
    Linear linear1, linear2, linear3;
    
    fcNetImpl(int input_size_, int output_size_, bool is_train_, float dprob_):
        input_size(input_size_),
        output_size(output_size_),
        is_train(is_train_),
        dprob(dprob_),
        linear1(register_module("linear1", Linear(LinearOptions(input_size, L1_UNIT).bias(true)))),
        linear2(register_module("linear2", Linear(LinearOptions(L1_UNIT, L2_UNIT).bias(true)))),
        linear3(register_module("linear3", Linear(LinearOptions(L2_UNIT, output_size).bias(true))))
    {
    }
    Tensor forward(Tensor x)
    {
        x = torch::dropout(x, dprob, is_train);
        x = torch::elu(linear1(x));

        x = torch::dropout(x, dprob, is_train);
        x = torch::elu(linear2(x));

        x = torch::dropout(x, dprob, is_train);
        x = linear3(x);

        return x;
    }
};

TORCH_MODULE(fcNet);

class fcnetImpl : public nnModule
{
public:
    int input_size, output_size;
    bool is_train;
    float dprob;
    Tensor W0, W1, W2, b0, b1, b2;

    fcnetImpl(int input_size_, int output_size_, bool is_train_, float dprob_):
        input_size(input_size_),
        output_size(output_size_),
        is_train(is_train_),
        dprob(dprob_),
        // 0번째 layer들 - 4 networks
        W0
        (
            register_parameter("W0", torch::randn({input_size, L1_UNIT}).mul(0.01f))
        ),
                               
        // 1번째 layer들 - 4 networks
        b0
        (
            register_parameter("b0", torch::zeros({L1_UNIT}))
        ),
        W1
        (
            register_parameter("W1", torch::randn({L1_UNIT, L2_UNIT}).mul(0.01f))
        ),
        b1
        (
            register_parameter("b1", torch::zeros({L2_UNIT}))
        ),
        // 2번째 layer들 - 4 networks
        W2
        (
            register_parameter("W2", torch::randn({L2_UNIT, output_size}).mul(0.01f))
        ),
        b2
        (
            register_parameter("b2", torch::zeros({output_size}))
        )
    {
    }
    Tensor forward(Tensor x)
    {
        //! it is used when using addmm
        if(x.dim() == 1)
        {
            x.unsqueeze_(0);
        }

        // x = torch::dropout(x, dprob, is_train);
        // x = torch::elu(torch::matmul(x, W0) + b0);
        x = torch::elu(torch::addmm(b0, x, W0));

        x = torch::dropout(x, dprob, is_train);
        // x = torch::elu(torch::matmul(x, W1) + b1);
        x = torch::elu(torch::addmm(b1, x, W1));

        // x = torch::dropout(x, dprob, is_train);
        // x = torch::matmul(x, W2) + b2;
        x = torch::addmm(b2, x, W2);
        return x;
    }
};

TORCH_MODULE(fcnet);


class PFNNImpl: public nnModule
{
public: 
    int nslice, input_size, output_size;
    bool is_train;
    float dprob;
    Tensor W0, W1, W2, b0, b1, b2;    //W, b parameters for each network
    Tensor W_[3], b_[3];                                      //최종 parameters for interpolated network


    PFNNImpl(int input_size_, int output_size_, bool is_train_, float dprob_):
        input_size(input_size_),
        output_size(output_size_),
        is_train(is_train_),
        dprob(dprob_),
        // 0번째 layer들 - 4 networks
        W0
        (
            register_parameter("W0", torch::randn({4, input_size, L1_UNIT}).mul(0.01f))
        ),
                               
        // 1번째 layer들 - 4 networks
        b0
        (
            register_parameter("b0", torch::zeros({4, 1, L1_UNIT}))
        ),
        W1
        (
            register_parameter("W1", torch::randn({4, L1_UNIT, L2_UNIT}).mul(0.01f))
        ),
        b1
        (
            register_parameter("b1", torch::zeros({4, 1, L2_UNIT}))
        ),
        // 2번째 layer들 - 4 networks
        W2
        (
            register_parameter("W2", torch::randn({4, L2_UNIT, output_size}).mul(0.01f))
        ),
        b2
        (
            register_parameter("b2", torch::zeros({4, 1, output_size}))
        )
    {

    }

    Tensor catmull_rom_spline(const Tensor& a0, const Tensor& a1, const Tensor& a2, const Tensor& a3, const Tensor& w_)
    {
        assert(("Tensor a0, a1, a2, a3 should have same sizes", 
                a0.sizes() == a1.sizes() && a0.sizes() == a2.sizes() && a0.sizes() == a3.sizes()));
        Tensor w = torch::clone(w_);
        w.detach_();
        
        for (int i = 0; i < a0.dim() -1; ++i)
        {
            w = w.unsqueeze(0);
        }
        w = torch::transpose(w, 0, w.dim() - 1);

        Tensor r1 = (a2.mul(0.5f) - a0.mul(0.5f)) * w;
        Tensor r2 = (a0 - a1.mul(2.5f) + a2.mul(2.0f) - a3.mul(0.5f))* w * w;
        Tensor r3 = (a1.mul(1.5f) - a2.mul(1.5f) + a3.mul(0.5f) - a0.mul(0.5f)) * w * w * w;

        return (a1 + r1 + r2 + r3);
    }

    Tensor forward(const Tensor& x, const Tensor& phases)
    {
        //Tensor x size: (batch_size X input_size)
        assert(("dimension of Tensor phases has to be 1", phases.dim() == 1));
        // w_ size: {batch_size} -> {batch_size x 1 x 1}
        Tensor w_ = torch::remainder((4 * phases) / (2 * M_PI), 1.0);
        Tensor pscale = torch::floor((4 * phases) / (2 * M_PI));

        Tensor k_0 = torch::remainder(pscale - 1, 4).toType(torch::kInt64);         // sizes(): inputSize
        Tensor k_1 = torch::remainder(pscale, 4).toType(torch::kInt64);             // sizes(): inputSize
        Tensor k_2 = torch::remainder(pscale + 1, 4).toType(torch::kInt64);         // sizes(): inputSize
        Tensor k_3 = torch::remainder(pscale + 2, 4).toType(torch::kInt64);         // sizes(): inputSize
        
        // size: {batchSize, input_size, L1_UNIT}
        Tensor W0_0 = torch::index_select(W0, 0, k_0);
        Tensor W0_1 = torch::index_select(W0, 0, k_1);
        Tensor W0_2 = torch::index_select(W0, 0, k_2);
        Tensor W0_3 = torch::index_select(W0, 0, k_3);

        // size: {batchSize, 1, L1_UNIT}
        Tensor b0_0 = torch::index_select(b0, 0, k_0);
        Tensor b0_1 = torch::index_select(b0, 0, k_1);
        Tensor b0_2 = torch::index_select(b0, 0, k_2);
        Tensor b0_3 = torch::index_select(b0, 0, k_3);

        // size: {batchSize, L1_UNIT, L2_UNIT}
        Tensor W1_0 = torch::index_select(W1, 0, k_0);
        Tensor W1_1 = torch::index_select(W1, 0, k_1);
        Tensor W1_2 = torch::index_select(W1, 0, k_2);
        Tensor W1_3 = torch::index_select(W1, 0, k_3);

        // size: {batchSize, 1, L2_UNIT}
        Tensor b1_0 = torch::index_select(b1, 0, k_0);
        Tensor b1_1 = torch::index_select(b1, 0, k_1);
        Tensor b1_2 = torch::index_select(b1, 0, k_2);
        Tensor b1_3 = torch::index_select(b1, 0, k_3);

        // size: {batchSize, L2_UNIT, output_size}
        Tensor W2_0 = torch::index_select(W2, 0, k_0);
        Tensor W2_1 = torch::index_select(W2, 0, k_1);
        Tensor W2_2 = torch::index_select(W2, 0, k_2);
        Tensor W2_3 = torch::index_select(W2, 0, k_3);

        // size: {batchSize, 1, output_size}
        Tensor b2_0 = torch::index_select(b2, 0, k_0);
        Tensor b2_1 = torch::index_select(b2, 0, k_1);
        Tensor b2_2 = torch::index_select(b2, 0, k_2);
        Tensor b2_3 = torch::index_select(b2, 0, k_3);

        W_[0] = catmull_rom_spline(W0_0, W0_1, W0_2, W0_3, w_);
        W_[1] = catmull_rom_spline(W1_0, W1_1, W1_2, W1_3, w_);
        W_[2] = catmull_rom_spline(W2_0, W2_1, W2_2, W2_3, w_);

        b_[0] = catmull_rom_spline(b0_0, b0_1, b0_2, b0_3, w_);
        b_[1] = catmull_rom_spline(b1_0, b1_1, b1_2, b1_3, w_);
        b_[2] = catmull_rom_spline(b2_0, b2_1, b2_2, b2_3, w_);

        int size =  3 - x.dim();
        Tensor new_x = torch::clone(x);

        for(int i =0; i < size; ++i)
        {
            new_x.unsqueeze_(0);
        }
        new_x.transpose_(0, 1);

        // layer1
        new_x = torch::dropout(new_x, dprob, is_train);
        new_x = torch::elu(torch::matmul(new_x, W_[0]) + b_[0]);  
        
        // layer2        
        new_x = torch::dropout(new_x, dprob, is_train);
        new_x = torch::elu(torch::matmul(new_x, W_[1]) + b_[1]);

        // layer3
        new_x = torch::dropout(new_x, dprob, is_train);
        new_x = torch::matmul(new_x, W_[2]) + b_[2];

        return new_x;
    }
};

TORCH_MODULE(PFNN);

}
