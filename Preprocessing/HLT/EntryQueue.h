/*! Definiton of a thread-safe fixed size entry queue. */

#pragma once

#include <mutex>
#include <queue>
#include <condition_variable>

namespace analysis {

template<typename _Entry>
class EntryQueue {
public:
  using Entry = _Entry;
  using Queue = std::queue<Entry>;
  using Mutex = std::mutex;
  using Lock = std::unique_lock<Mutex>;
  using CondVar = std::condition_variable;

public:
  explicit EntryQueue(size_t max_size, size_t max_entries = std::numeric_limits<size_t>::max())
    : max_size_(max_size), max_entries_(max_entries), n_entries_(0), input_depleted_(false), output_depleted_(false)
  {
  }

  bool Push(const Entry& entry)
  {
    bool entry_is_pushed = false;
    {
      Lock lock(mutex_);
      cond_var_.wait(lock, [&] { return queue_.size() < max_size_ || n_entries_ >= max_entries_ || output_depleted_; });
      if(!(n_entries_ >= max_entries_ || input_depleted_)) {
        queue_.push(entry);
        ++n_entries_;
        entry_is_pushed = true;
      }
    }
    cond_var_.notify_all();
    return entry_is_pushed;
  }

  bool Pop(Entry& entry)
  {
    bool entry_is_valid = false;
    {
      Lock lock(mutex_);
      cond_var_.wait(lock, [&] { return !queue_.empty() || input_depleted_; });
      if(!queue_.empty()) {
        entry = queue_.front();
        entry_is_valid = true;
        queue_.pop();
      }
    }
    cond_var_.notify_all();
    return entry_is_valid;
  }

  void SetInputDepleted()
  {
    {
      Lock lock(mutex_);
      input_depleted_ = true;
    }
    cond_var_.notify_all();
  }

  void SetOutputDepleted()
  {
    {
      Lock lock(mutex_);
      output_depleted_ = true;
    }
    cond_var_.notify_all();
  }

private:
  Queue queue_;
  const size_t max_size_, max_entries_;
  size_t n_entries_;
  bool input_depleted_, output_depleted_;
  Mutex mutex_;
  CondVar cond_var_;
};

} // namespace analysis
