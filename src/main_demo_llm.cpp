//
// Created by 69029 on 4/12/2021.
//

#undef NDEBUG
#include "circuit.h"
#include "neuralNetwork.hpp"
#include "verifier.hpp"
#include "models.hpp"
#include "global_var.hpp"
#include <iostream>

using namespace mcl::bn;
using namespace std;




int main(int argc, char **argv) 
{
    initPairing(mcl::BN254);
    prover p;
    LLM nn(12);  // GPT-2 has 12 transformer blocks
    nn.create(p, 1);
    verifier v(&p, p.C);
    v.prove(32); // prove with 32 threads
}

