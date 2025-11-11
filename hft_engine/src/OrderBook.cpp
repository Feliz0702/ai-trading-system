#include "OrderBook.h"
#include <algorithm>
#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
#endif

OrderBook::OrderBook(const std::string& symbol)
    : symbol_(symbol)
    , bids_(MAX_DEPTH_LEVELS)
    , asks_(MAX_DEPTH_LEVELS)
    , order_pool_(INITIAL_ORDER_POOL_SIZE)
    , trade_pool_(INITIAL_TRADE_POOL_SIZE)
    , incoming_orders_(LOCK_FREE_QUEUE_SIZE)
{
    order_pool_.preallocate();
    trade_pool_.preallocate();
}

OrderBook::~OrderBook() = default;

MatchingResult OrderBook::add_order(Order&& order){
    Order* ptr = order_pool_.acquire();
    *ptr = std::move(order);
    ptr->timestamp = get_nanoseconds();
    std::atomic_thread_fence(std::memory_order_acq_rel);
    if(!incoming_orders_.try_push(ptr)){
        order_pool_.release(ptr);
        return {false, "Order queue full"};
    }
    return {true, "Order accepted"};
}

bool OrderBook::cancel_order(uint64_t order_id){
    auto it = order_index_.find(order_id);
    if(it==order_index_.end()) return false;
    Order* o = it->second; o->quantity = 0.0; // mark cancelled
    order_index_.erase(it);
    return true;
}

bool OrderBook::modify_order(uint64_t order_id, double new_quantity){
    auto it = order_index_.find(order_id);
    if(it==order_index_.end()) return false;
    it->second->quantity = new_quantity; return true;
}

OrderBookSnapshot OrderBook::get_snapshot() const{
    OrderBookSnapshot s; s.bids = bids_; s.asks = asks_; s.sequence = sequence_.load(); return s;
}

void OrderBook::matching_cycle(){
    constexpr int BATCH=16; Order* batch[BATCH]; int processed=0; Order* tmp=nullptr;
    while(processed < BATCH && incoming_orders_.try_pop(tmp)) batch[processed++]=tmp;
    if(processed>0) process_order_batch(batch, processed);
    match_orders();
}

void OrderBook::process_order_batch(Order** orders, int count){
    // 可選使用 SIMD 讀取部分欄位，這裡僅保留接口以便後續擴展
    for(int i=0;i<count;++i){ add_order_to_book(orders[i]); }
}

void OrderBook::add_order_to_book(Order* order){
    order_index_[order->order_id] = order;
    auto& book = (order->side==Side::BUY) ? bids_ : asks_;
    // 將數量累加到最接近的價位（簡化：直接取索引），避免宏干擾使用括號調用
    int price_index = static_cast<int>(order->price) % MAX_DEPTH_LEVELS;
    if(price_index < 0) price_index += MAX_DEPTH_LEVELS;
    int idx = (std::min)(MAX_DEPTH_LEVELS - 1, (std::max)(0, price_index));
    book[idx].price = order->price;
    book[idx].quantity += order->quantity;
    sequence_.fetch_add(1, std::memory_order_relaxed);
}

void OrderBook::match_orders(){
    // 極簡撮合：尋找最高買價與最低賣價，若交叉則成交最小量
    double best_bid=-1e300; int bid_idx=-1;
    for(int i=0;i<MAX_DEPTH_LEVELS;++i){ if(bids_[i].quantity>0 && bids_[i].price>best_bid){ best_bid=bids_[i].price; bid_idx=i; } }
    double best_ask=1e300; int ask_idx=-1;
    for(int i=0;i<MAX_DEPTH_LEVELS;++i){ if(asks_[i].quantity>0 && asks_[i].price<best_ask){ best_ask=asks_[i].price; ask_idx=i; } }
    if(bid_idx>=0 && ask_idx>=0 && best_bid >= best_ask){
        double qty = std::min(bids_[bid_idx].quantity, asks_[ask_idx].quantity);
        bids_[bid_idx].quantity -= qty; asks_[ask_idx].quantity -= qty;
        // 建立交易（僅統計，不返回）
        Trade* t = trade_pool_.acquire(); t->price = (best_bid+best_ask)/2.0; t->quantity = qty; t->timestamp = get_nanoseconds();
        (void)t; // 在此簡化：不放入外部佇列
        sequence_.fetch_add(1, std::memory_order_relaxed);
    }
}

void OrderBook::process_trade(Order* /*taker*/, Order* /*maker*/, double /*quantity*/){
    // 詳細撮合留作後續擴展
}
