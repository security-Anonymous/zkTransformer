//
// Created by 69029 on 3/25/2021.
//

#ifndef ZKCNN_TIMER_HPP
#define ZKCNN_TIMER_HPP

#include <chrono>
#include <assert.h>

class timer {
public:
    timer(bool accumulate_=false) { total_time_sec = 0; status = false; accumulate=accumulate_;}
    void start();
    void stop(const char* out,bool total=true,bool restart=true);
    void stop();
    void clear() { total_time_sec = 0; status = false;}
    double elapse_sec() const 
    {
        assert(status == false);
        return total_time_sec;
    }
private:
    std::chrono::high_resolution_clock::time_point t0;
    double total_time_sec;
    bool status;
    bool accumulate;
};


#endif //ZKCNN_TIMER_HPP