#include <cassert>
#include "ObjectPool.h"

struct Foo{ int a{0}; Foo()=default; ~Foo()=default; };

int main(){
    ObjectPool<Foo> pool(128);
    pool.preallocate();
    Foo* f = pool.acquire();
    f->a = 42;
    pool.release(f);
    Foo* f2 = pool.acquire();
    // object reset by placement new
    assert(f2->a==0);
    pool.release(f2);
    return 0;
}
