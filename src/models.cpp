//
// Created by 69029 on 3/16/2021.
//

#include <tuple>
#include <iostream>
#include "models.hpp"
#include "utils.hpp"
#undef USE_VIRGO




LLM::LLM(int depth):
neuralNetwork(0,0,0,0, "./data/vgg11/vgg11.cifar.relu-1-images-weights-qint8.csv", "", "",true)
{
    len=30;
    pic_parallel=1;
    conv_section.clear();
    positive_check=0;
    exp_check=0;
    layer_num=depth;
    for(int i=0;i<depth;i++)
    {
        full_conn.emplace_back(2304,768);  //now debugging which act made trouble
        full_conn.emplace_back(768,768);
        full_conn.emplace_back(2304,768);
        full_conn.emplace_back(768,2304);
    }
    
}