#ifndef ZKGPT_HPP
#define ZKGPT_HPP

#include "neuralNetwork.hpp"



class LLM: public neuralNetwork {

public:
    explicit LLM(int depth);

};

#endif 
