#include <cassert>
#include <iostream>
#include "OrderBook.h"

static void pump_cycles(OrderBook& book, int n){ for(int i=0;i<n;++i) book.matching_cycle(); }

int main(){
    std::cout << "=== Order Ops Tests ===\n";
    OrderBook book("BTC/USDT");

    // add one order
    Order o{}; o.order_id=123; o.symbol="BTC/USDT"; o.side=Side::BUY; o.type=OrderType::LIMIT; o.price=50000.0; o.quantity=1.0;
    auto res = book.add_order(std::move(o));
    assert(res.accepted || res.success); // support both MatchingResult shapes
    pump_cycles(book, 2);

    // modify
    bool m = book.modify_order(123, 0.5);
    assert(m);
    pump_cycles(book, 2);

    // cancel
    bool c = book.cancel_order(123);
    assert(c);
    pump_cycles(book, 2);

    std::cout << "PASS\n";
    return 0;
}
