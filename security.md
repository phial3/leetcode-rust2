Rust 并发、安全和异步变成高性能库

 1. 并行计算：
- [rayon:数据并行库](https://github.com/rayon-rs/rayon)
- [Crossbeam:高级工具](https://github.com/crossbeam-rs/crossbeam)
2. 同步原语：
- [parking_lot:高效锁](https://github.com/Amanieu/parking_lot)
- [spin：提供自旋锁，延迟初始化等轻量级同步原语，无操作系统依赖](https://github.com/mvdnes/spin-rs)
- [atomic-rs:扩展标准库的原子操作](https://github.com/Amanieu/atomic-rs)
- [thread_local](https://github.com/Amanieu/thread_local-rs)
3. 异步运行：
- [Tokio:异步运行时](https://github.com/tokio-rs/tokio)
- [async-std:异步标准库](https://github.com/async-rs/async-std)
- [smol:轻量级异步运行时](https://github.com/smol-rs/smol)
4. 异步底层：
- [features:异步编程的基础库](https://github.com/rust-lang/futures-rs)
- [mio:底层非阻塞 IO 库](https://github.com/tokio-rs/mio)
- [polling:轻量级，替代 mio 的简化版](https://github.com/smol-rs/polling)
5. 通道与消息传递：
- [Flume:支持同步和异步通道](https://github.com/zesterer/flume)
- [concurrent-queue:多种并发队列（无锁队列、阻塞队列）](https://github.com/smol-rs/concurrent-queue)
- Crossbeam-channel: Crossbeam 的一部分
6. 并发数据结构：
- [DashMap:并发哈希表](https://github.com/xacrimon/dashmap)
- [arc-swap:高效原子引用计数 Arc 的替代方案](https://github.com/vorner/arc-swap)
- crossbeam-epoch: Crossbeam 的一部分,基于 Epoch 的内存回收机制，用于构建无锁数据结构
7. 核心库：
- [lazy_static:延迟初始化全局静态变量,多线程安全访问](https://github.com/rust-lang-nursery/lazy-static.rs)
- [once_cell:单次初始化,线程安全且高效](https://github.com/matklad/once_cell)
8. 测试与调试：
- [Loom:并发代码测试工具](https://github.com/tokio-rs/loom)

