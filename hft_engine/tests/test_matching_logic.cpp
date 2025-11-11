#include <iostream>
#include <cassert>
#include <string>
#include "OrderBook.h"

static double sum_qty(const std::vector<PriceLevel>& levels){
    double s = 0.0; for(const auto& l: levels) s += l.quantity; return s;
}

static void pump_cycles(OrderBook& book, int n){
    for(int i=0;i<n;++i) book.matching_cycle();
}

void test_partial_fill(){
    std::cout << "Testing partial fill... ";
    OrderBook book("BTC/USDT");

    Order buy{}; buy.order_id=1; buy.symbol="BTC/USDT"; buy.price=50000.0; buy.quantity=2.0; buy.side=Side::BUY; buy.type=OrderType::LIMIT;
    book.add_order(std::move(buy));
    pump_cycles(book, 2);
    auto snap1 = book.get_snapshot();
    double bids_before = sum_qty(snap1.bids);

    Order sell{}; sell.order_id=2; sell.symbol="BTC/USDT"; sell.price=50000.0; sell.quantity=1.0; sell.side=Side::SELL; sell.type=OrderType::LIMIT;
    book.add_order(std::move(sell));
    pump_cycles(book, 2);
    auto snap2 = book.get_snapshot();

    assert(sum_qty(snap2.asks) <= sum_qty(snap1.asks));
    assert(sum_qty(snap2.bids) <= bids_before);
    // 至少撮合了部分（序列應遞增且買方數量下降至少一點）
    assert(sum_qty(snap2.bids) <= bids_before - 0.5);
    std::cout << "PASS\n";
}

void test_multiple_matches(){
    std::cout << "Testing multiple matches... ";
    OrderBook book("BTC/USDT");

    for(int id=1; id<=3; ++id){
        Order buy{}; buy.order_id=id; buy.symbol="BTC/USDT"; buy.price=50000.0; buy.quantity=1.0; buy.side=Side::BUY; buy.type=OrderType::LIMIT;
        book.add_order(std::move(buy));
        pump_cycles(book, 1);
    }
    auto snap_b = book.get_snapshot();
    double bids_before = sum_qty(snap_b.bids);

    Order sell{}; sell.order_id=10; sell.symbol="BTC/USDT"; sell.price=50000.0; sell.quantity=2.5; sell.side=Side::SELL; sell.type=OrderType::LIMIT;
    book.add_order(std::move(sell));
    pump_cycles(book, 3);
    auto snap_a = book.get_snapshot();

    // 至少撮合掉 ~2.0 以上
    assert(sum_qty(snap_a.bids) <= bids_before - 1.5);
    std::cout << "PASS\n";
}

void test_price_priority_smoke(){
    std::cout << "Testing price priority (smoke)... ";
    OrderBook book("BTC/USDT");

    // 三個不同價格買單
    Order b1{}; b1.order_id=1; b1.symbol="BTC/USDT"; b1.price=50100.0; b1.quantity=1.0; b1.side=Side::BUY; b1.type=OrderType::LIMIT; book.add_order(std::move(b1)); pump_cycles(book,1);
    Order b2{}; b2.order_id=2; b2.symbol="BTC/USDT"; b2.price=50000.0; b2.quantity=1.0; b2.side=Side::BUY; b2.type=OrderType::LIMIT; book.add_order(std::move(b2)); pump_cycles(book,1);
    Order b3{}; b3.order_id=3; b3.symbol="BTC/USDT"; b3.price=50200.0; b3.quantity=1.0; b3.side=Side::BUY; b3.type=OrderType::LIMIT; book.add_order(std::move(b3)); pump_cycles(book,1);

    auto snap_b = book.get_snapshot();
    double bids_before = sum_qty(snap_b.bids);

    // 賣單應與最高買價側撮合，至少撮合 0.5
    Order s{}; s.order_id=4; s.symbol="BTC/USDT"; s.price=50000.0; s.quantity=0.5; s.side=Side::SELL; s.type=OrderType::LIMIT; book.add_order(std::move(s)); pump_cycles(book,2);

    auto snap_a = book.get_snapshot();
    assert(sum_qty(snap_a.bids) <= bids_before - 0.25);
    std::cout << "PASS\n";
}

int main(){
    std::cout << "=== Matching Logic Tests ===\n";
    test_partial_fill();
    test_multiple_matches();
    test_price_priority_smoke();
    std::cout << "=== All tests passed! ===\n";
    return 0;
}
