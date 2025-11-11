#include <cassert>
#include "OrderBook.h"

int main(){
    OrderBook ob("TEST");
    Order o; o.order_id=1; o.symbol="TEST"; o.side=Side::BUY; o.type=OrderType::LIMIT; o.price=100.0; o.quantity=1.5;
    auto res = ob.add_order(std::move(o));
    assert(res.accepted);
    ob.matching_cycle();
    auto snap = ob.get_snapshot();
    // either quantity recorded on some level or zero — smoke check
    bool any=false; for(const auto& pl: snap.bids){ if(pl.quantity>0){ any=true; break; } }
    assert(any);
    return 0;
}
