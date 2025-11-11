#pragma once
#include <string>
#include "Types.h"

class FPGAInterface {
public:
    bool initialize(const std::string& /*device_path*/){ return false; }
    bool send_order_to_fpga(const Order& /*order*/){ return false; }
    bool read_match_result_from_fpga(Trade& /*trade*/){ return false; }
};
