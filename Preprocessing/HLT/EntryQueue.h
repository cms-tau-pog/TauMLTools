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

struct QueueState {
  size_t max_size, max_entries, n_entries, waiting_for_entry, n_max_size_reached, n_zero_size_reached, queue_size;
  bool input_depleted, output_depleted;

  QueueState(size_t max_size_, size_t max_entries_)
    : max_size(max_size_), max_entries(max_entries_), n_entries(0), waiting_for_entry(0), n_max_size_reached(0),
      n_zero_size_reached(0), queue_size(0), input_depleted(false), output_depleted(false)
  {
  }
};

template<typename _Entry>
class MultiSlotEntryQueue {
public:
  using Entry = _Entry;
  using Queue = std::queue<Entry>;
  using Mutex = std::mutex;
  using Lock = std::unique_lock<Mutex>;
  using CondVar = std::condition_variable;

  struct QueueDesc {
    QueueState state;
    Queue queue;
    QueueDesc(size_t max_size_, size_t max_entries_) : state(max_size_, max_entries_) {}
  };

  enum class PushResult {
    EntryPushed, MaxSizeReached, MaxEntriesReached, OutputDepleted,
  };

public:
  size_t AddQueue(size_t max_size, size_t max_entries = std::numeric_limits<size_t>::max())
  {
    size_t slot_id;
    {
      Lock lock(mutex_);
      queues_.emplace_back(std::make_unique<QueueDesc>(max_size, max_entries));
      slot_id = queues_.size() - 1;
    }
    cond_var_.notify_all();
    return slot_id;
  }

  PushResult Push(size_t slot_id, const Entry& entry)
  {
    PushResult push_result;
    {
      Lock lock(mutex_);
      CheckSlotId(slot_id);
      cond_var_.wait(lock, [&] {
        const auto& desc = *queues_.at(slot_id);
        const auto& state = desc.state;
        if(desc.queue.size() < state.max_size || state.n_entries >= state.max_entries || state.output_depleted)
          return true;
        for(const auto& queue_desc : queues_) {
          if(queue_desc->state.waiting_for_entry > 0)
            return true;
        }
        return false;
      });
      auto& desc = *queues_.at(slot_id);
      auto& state = desc.state;
      if(desc.queue.size() >= state.max_size) {
        push_result = PushResult::MaxSizeReached;
        ++state.n_max_size_reached;
      } else if(state.n_entries >= state.max_entries) {
        push_result = PushResult::MaxEntriesReached;
      } else if(state.output_depleted) {
        push_result = PushResult::OutputDepleted;
      } else {
        if(desc.queue.empty())
          ++state.n_zero_size_reached;
        desc.queue.push(entry);
        ++state.n_entries;
        push_result = PushResult::EntryPushed;
      }
    }
    cond_var_.notify_all();
    return push_result;
  }

  bool Pop(size_t slot_id, Entry& entry)
  {
    bool entry_is_valid = false;
    {
      Lock lock(mutex_);
      CheckSlotId(slot_id);
      ++queues_.at(slot_id)->state.waiting_for_entry;
    }
    cond_var_.notify_all();
    {
      Lock lock(mutex_);
      cond_var_.wait(lock, [&] {
        return !queues_.at(slot_id)->queue.empty() || queues_.at(slot_id)->state.input_depleted;
      });
      auto& desc = *queues_.at(slot_id);
      if(!desc.queue.empty()) {
        entry = desc.queue.front();
        entry_is_valid = true;
        desc.queue.pop();
        --desc.state.waiting_for_entry;
      }
    }
    cond_var_.notify_all();
    return entry_is_valid;
  }

  void SetInputDepleted(size_t slot_id)
  {
    {
      Lock lock(mutex_);
      CheckSlotId(slot_id);
      queues_.at(slot_id)->state.input_depleted = true;
    }
    cond_var_.notify_all();
  }

  void SetInputDepleted()
  {
    {
      Lock lock(mutex_);
      for(auto& desc : queues_)
        desc->state.input_depleted = true;
    }
    cond_var_.notify_all();
  }


  void SetOutputDepleted(size_t slot_id)
  {
    {
      Lock lock(mutex_);
      CheckSlotId(slot_id);
      queues_.at(slot_id)->state.output_depleted = true;
    }
    cond_var_.notify_all();
  }

  void SetOutputDepleted()
  {
    {
      Lock lock(mutex_);
      for(auto& desc : queues_)
        desc->state.output_depleted = true;
    }
    cond_var_.notify_all();
  }

  bool IsMaxEntriesReachedForAll()
  {
    Lock lock(mutex_);
    for(const auto& desc : queues_) {
      if(desc->state.n_entries < desc->state.max_entries)
        return false;
    }
    return true;
  }

  QueueState GetQueueState(size_t slot_id)
  {
    Lock lock(mutex_);
    CheckSlotId(slot_id);
    QueueState state = queues_.at(slot_id)->state;
    state.queue_size = queues_.at(slot_id)->queue.size();
    return state;
  }

private:
  void CheckSlotId(size_t slot_id) const
  {
    if(slot_id >= queues_.size())
      throw std::runtime_error("MultiSlotEntryQueue: slot ID out of range");
  }

private:
  std::vector<std::unique_ptr<QueueDesc>> queues_;
  Mutex mutex_;
  CondVar cond_var_;
};

inline std::ostream& operator<<(std::ostream& out, const QueueState& state)
{
  out << "max_size=" << state.max_size << ", max_entries=" << state.max_entries << ", n_entries=" << state.n_entries
      << ", waiting_for_entry=" << state.waiting_for_entry << ", n_max_size_reached=" << state.n_max_size_reached
      << ", n_zero_size_reached=" << state.n_zero_size_reached << ", queue_size=" << state.queue_size
      << ", input_depleted=" << state.input_depleted << ", output_depleted=" << state.output_depleted;
  return out;
}

} // namespace analysis
