#pragma once

#include <map>
#include <vector>
#include <thread>
#include <typeinfo>
#include <variant>

#include "EntryQueue.h"

namespace analysis {

struct Entry {
  bool valid{false};
  using Variant = std::variant<int, float, RVecI, RVecF>;
  std::vector<Variant> values;

  explicit Entry(size_t size) : values(size) {}

  template<typename T>
  void Set(int index, const T& value)
  {
    if (index >= values.size())
      throw std::runtime_error("Entry::Add: index out of range");
    values[index] = value;
  }
};

struct StopLoop {};

struct BinDesc {
  size_t slot_id;
  int start_idx, stop_idx;

  bool IsInside(ULong64_t rdfentry, int batch_size) const
  {
    const int index = rdfentry % batch_size;
    return index >= start_idx && index < stop_idx;
  }
};

namespace detail {
inline void putEntry(std::vector<std::shared_ptr<Entry>>& entries, int index) {}

template<typename T, typename ...Args>
void putEntry(std::vector<std::shared_ptr<Entry>>& entries, int var_index, const ROOT::VecOps::RVec<T>& value,
              Args&& ...args)
{
  if(value.size() != entries.size())
    throw std::runtime_error("TupleMaker::putEntry: vector size mismatch");
  for(size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    entries[entry_index]->Set(var_index, value[entry_index]);
  putEntry(entries, var_index + 1, std::forward<Args>(args)...);
}

inline std::string timeStamp(const std::chrono::system_clock::time_point& time_point)
{
    std::ostringstream ss;
    const std::time_t time_t_point = std::chrono::system_clock::to_time_t(time_point);
    //ss << "[" << std::put_time(std::localtime(&time_t_point), "%F %T") << "] ";
    char mbstr[100];
    if (std::strftime(mbstr,sizeof(mbstr),"%F %T",std::localtime(&time_t_point)))
        ss << "[" << mbstr << "] ";
    return ss.str();
}

inline std::string timeStampNow()
{
    return timeStamp(std::chrono::system_clock::now());
}

inline int getDuration(const std::chrono::system_clock::time_point& start,
                       const std::chrono::system_clock::time_point& end)
{
  const auto delta_t = end - start;
  const auto duration = std::chrono::duration_cast<std::chrono::seconds>(delta_t).count();
  return static_cast<int>(duration);
}

} // namespace detail

template<typename ...Args>
struct TupleMaker {
  static constexpr size_t nArgs = sizeof...(Args);
  using Queue = MultiSlotEntryQueue<std::shared_ptr<Entry>>;
  using PushResult = Queue::PushResult;

  TupleMaker() {}
  TupleMaker(const TupleMaker&) = delete;
  TupleMaker& operator= (const TupleMaker&) = delete;

  int AddBin(int start_idx, int stop_idx, size_t max_queue_size, size_t max_entries)
  {
    const size_t slot_id = queue.AddQueue(max_queue_size, max_entries);
    bins.emplace_back(BinDesc{ slot_id, start_idx, stop_idx});
    return slot_id;
  }

  bool GetSlotId(ULong64_t rdfentry, int batch_size, size_t& slot_id) const
  {
    for(const auto& bin : bins) {
      if(bin.IsInside(rdfentry, batch_size)) {
        slot_id = bin.slot_id;
        return true;
      }
    }
    return false;
  }

  ROOT::RDF::RNode process(ROOT::RDF::RNode df_in, ROOT::RDF::RNode df_out, const std::vector<std::string>& var_names,
                           int batch_size, bool infinite_input)
  {
    thread = std::make_unique<std::thread>([=]() {
      std::cout << "TupleMaker::process: foreach started." << std::endl;
      ROOT::RDF::RNode df = df_in;
      bool do_next_iteration = true;
      for(size_t iteration_number = 1; do_next_iteration; ++iteration_number) {
        try {
          std::cout << detail::timeStampNow() << "TupleMaker::process: foreach iteration " << iteration_number
                    << std::endl;
          df.Foreach([&](const RVecI& slot_sel, const Args& ...args) {
            std::vector<std::shared_ptr<Entry>> entries(slot_sel.size());
            for(size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
              entries[entry_index] = std::make_shared<Entry>(nArgs);
            detail::putEntry(entries, 0, args...);
            for(size_t entry_index = 0; entry_index < slot_sel.size(); ++entry_index) {
              if(slot_sel[entry_index] < 0) continue;
              const size_t slot_id = slot_sel[entry_index];
              auto& entry = entries[entry_index];
              entry->valid = true;
              // std::cout << "Push to slot " << slot_id << " started." << std::endl;
              const auto push_result = queue.Push(slot_id, entry);
              // std::cout << "Push to slot " << slot_id << " done." << std::endl;
              if(queue.IsMaxEntriesReachedForAll())
                throw StopLoop();
            }
          }, var_names);
        } catch(StopLoop) {
          std::cout << detail::timeStampNow() << "TupleMaker::process: queue output is depleted." << std::endl;
          do_next_iteration = false;
        }
        if(!infinite_input)
          do_next_iteration = false;
        std::set<size_t> empty_bins;
        for(size_t bin_idx = 0; bin_idx < bins.size(); ++bin_idx) {
          const auto& desc = bins[bin_idx];
          const auto queue_state = queue.GetQueueState(desc.slot_id);
          if(queue_state.n_entries == 0)
            empty_bins.insert(bin_idx);
          std::cout << "bin " << bin_idx << ": " << queue_state << std::endl;
        }
        if(!empty_bins.empty()) {
          std::cout << "TupleMaker::process: empty bins: ";
          for(const auto& bin_idx : empty_bins)
            std::cout << bin_idx << " ";
          std::cout << std::endl;
          throw std::runtime_error("TupleMaker::process: empty bins after processing the full dataset");
        }
      }
      queue.SetInputDepleted();
      std::cout << "TupleMaker::process: foreach done." << std::endl;
    });
    df_out = df_out.Define("_entry", [=](ULong64_t rdfentry) {
      std::shared_ptr<Entry> entry;
      size_t slot_id;
      if(GetSlotId(rdfentry, batch_size, slot_id)) {
        // std::cout << "Pop from slot " << slot_id << " started." << std::endl;
        const auto pop_start = std::chrono::system_clock::now();
        queue.Pop(slot_id, entry);
        const auto duration = detail::getDuration(pop_start, std::chrono::system_clock::now());
        if(duration > 60)
          std::cout << detail::timeStampNow() << "Pop took " << duration
                    << " seconds to get the next entry for slot " << slot_id << std::endl;
      }
      const size_t batch_idx = rdfentry / batch_size;
      if((rdfentry % batch_size == 0) && (batch_idx % 100 == 0))
        std::cout << detail::timeStampNow() << "Processed " << batch_idx  << " batches" << std::endl;
      return entry;
    }, { "rdfentry_" });
    return df_out;
  }

  void join()
  {
    if(thread)
      thread->join();
  }

  Queue queue;
  std::vector<BinDesc> bins;
  std::unique_ptr<std::thread> thread;
};

} // namespace analysis