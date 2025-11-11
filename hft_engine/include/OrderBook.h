#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>
#include "Types.h"
#include "ObjectPool.h"
#include "LockFreeQueue.h"

class OrderBook {
private:
    std::string symbol_;
    std::atomic<uint64_t> sequence_{0};
    std::vector<PriceLevel> bids_;
    std::vector<PriceLevel> asks_;
    std::unordered_map<uint64_t, Order*> order_index_;
    ObjectPool<Order> order_pool_;
    ObjectPool<Trade> trade_pool_;
    LockFreeQueue<Order*> incoming_orders_;
public:
    explicit OrderBook(const std::string& symbol);
    ~OrderBook();

    MatchingResult add_order(Order&& order);
    bool cancel_order(uint64_t order_id);
    bool modify_order(uint64_t order_id, double new_quantity);

    const std::vector<PriceLevel>& get_bids() const { return bids_; }
    const std::vector<PriceLevel>& get_asks() const { return asks_; }
    OrderBookSnapshot get_snapshot() const;

    void matching_cycle();
private:
    void add_order_to_book(Order* order);
    void match_orders();
    void process_trade(Order* taker, Order* maker, double quantity);
    void process_order_batch(Order** orders, int count);
};
