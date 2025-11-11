#include <cassert>
#include <thread>
#include <vector>
#include "LockFreeQueue.h"

int main(){
    LockFreeQueue<int> q(1024);
    // single producer/consumer smoke
    for(int i=0;i<1000;++i){ bool ok = q.try_push(i); assert(ok); }
    int x; int cnt=0; while(q.try_pop(x)){ assert(x>=0); ++cnt; }
    assert(cnt==1000);
    // concurrent
    LockFreeQueue<int> q2(4096);
    std::thread t1([&]{ for(int i=0;i<10000;++i){ while(!q2.try_push(i)){} } });
    int sum=0; std::thread t2([&]{ int v; int c=0; while(c<10000){ if(q2.try_pop(v)){ sum+=v; ++c; } } });
    t1.join(); t2.join();
    assert(sum>=0);
    return 0;
}
