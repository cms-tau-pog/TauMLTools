/*! Definiton of a thread-safe fixed size entry queue. */

#pragma once

#include <mutex>
#include <queue>
#include <condition_variable>

namespace analysis {

template<typename Entry>
class EntryQueue {
public:
  using Queue = std::queue<Entry>;
  using Mutex = std::mutex;
  using Lock = std::unique_lock<Mutex>;
  using CondVar = std::condition_variable;

public:
  explicit EntryQueue(size_t max_size, size_t max_entries = std::numeric_limits<size_t>::max())
    : max_size_(max_size), max_entries_(max_entries), n_entries_(0), all_done_(false)
  {
  }

  bool Push(const Entry& entry)
  {
    {
      Lock lock(mutex_);
      if(n_entries_ >= max_entries_)
        return false;
      cond_var_.wait(lock, [&] { return queue_.size() < max_size_; });
      queue_.push(entry);
      ++n_entries_;
    }
    cond_var_.notify_all();
    return true;
  }

  bool Pop(Entry& entry)
  {
    bool entry_is_valid = false;;
    {
      Lock lock(mutex_);
      cond_var_.wait(lock, [&] { return queue_.size() || all_done_; });
      if(!queue_.empty()) {
        entry = queue_.front();
        entry_is_valid = true;
        queue_.pop();
      }
    }
    cond_var_.notify_all();
    return entry_is_valid;
  }

  void SetAllDone(bool value = true)
  {
    {
      Lock lock(mutex_);
      all_done_ = value;
    }
    cond_var_.notify_all();
  }

private:
  Queue queue_;
  const size_t max_size_, max_entries_;
  size_t n_entries_;
  bool all_done_;
  Mutex mutex_;
  CondVar cond_var_;
};

} // namespace analysis
