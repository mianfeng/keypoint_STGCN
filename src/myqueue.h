#ifndef MYQUEUE_H
#define MYQUEUE_H

#include <unistd.h>
// 并行
#include <condition_variable>
#include <queue>
#include <thread>
// 互斥访问
#include <atomic>
#include <mutex>

template <typename T>
class AysncQueue {
   public:
    // AsyncQueue(){}

    /**
     * 入队列
     */
    void enqueue(T val) {
        std::unique_lock<std::mutex> lock(mMutex);
        mQueue.push(std::move(val));
        mCond.notify_all();
    }

    /**
     * 出队列
     */
    T dequeue() {
        std::unique_lock<std::mutex> lock(mMutex);
        while (mQueue.empty()) {
            mCond.wait(lock);
        }
        auto val = std::move(mQueue.front());
        mQueue.pop();
        return val;
    }

    int size() {
        std::unique_lock<std::mutex> lock(mMutex);
        return mQueue.size();
    }

   private:
    std::mutex mMutex;
    std::condition_variable mCond;
    std::queue<T> mQueue;
};

#endif