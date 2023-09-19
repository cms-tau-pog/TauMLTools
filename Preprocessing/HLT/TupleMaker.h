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
  std::map<int, Variant> values;

  template<typename T>
  void Add(int index, const T& value)
  {
    if (values.count(index))
      throw std::runtime_error("Entry::Add: index already exists");
    values[index] = value;
  }
};

struct StopLoop {};

namespace detail {
inline void putEntry(std::vector<Entry>& entries, int index, const RVecB& sel) {}

template<typename T, typename ...Args>
void putEntry(std::vector<Entry>& entries, int var_index, const RVecB& sel,
              const ROOT::VecOps::RVec<T>& value, Args&& ...args)
{
  if(entries.empty()) {
    size_t n = 0;
    for(size_t i = 0; i < sel.size(); ++i) {
      if(sel[i])
        ++n;
    }
    entries.resize(n);
  }
  for(size_t entry_index = 0, out_index = 0; entry_index < value.size(); ++entry_index) {
    if(sel.at(entry_index)) {
      entries.at(out_index).Add(var_index, value[entry_index]);
      ++out_index;
    }
  }
  putEntry(entries, var_index + 1, sel, std::forward<Args>(args)...);
}

} // namespace detail

template<typename ...Args>
struct TupleMaker {
  TupleMaker(size_t queue_size, size_t max_entries)
    : queue(queue_size, max_entries)
  {
  }

  TupleMaker(const TupleMaker&) = delete;
  TupleMaker& operator= (const TupleMaker&) = delete;

  ROOT::RDF::RNode process(ROOT::RDF::RNode df_in, ROOT::RDF::RNode df_out, const std::vector<std::string>& var_names,
                           int start_idx, int stop_idx, int batch_size, bool infinite_input)
  {
    thread = std::make_unique<std::thread>([=]() {
      std::cout << "TupleMaker::process: foreach started." << std::endl;
      ROOT::RDF::RNode df = df_in;
      bool do_next_iteration = true;
      for(size_t iteration_number = 1; do_next_iteration; ++iteration_number) {
        try {
          std::cout << "TupleMaker::process: foreach iteration " << iteration_number << std::endl;
          df.Foreach([&](const Args& ...args) {
            std::vector<Entry> entries;
            detail::putEntry(entries, 0, args...);
            for(auto& entry : entries) {
              entry.valid = true;
              if(!queue.Push(entry))
                throw StopLoop();
            }
          }, var_names);
        } catch(StopLoop) {
          std::cout << "TupleMaker::process: queue output is depleted." << std::endl;
          do_next_iteration = false;
        }
        if(!infinite_input)
          do_next_iteration = false;
      }
      queue.SetOutputDepleted();
      std::cout << "TupleMaker::process: foreach done." << std::endl;
    });
    df_out = df_out.Define("_entry", [=](ULong64_t rdfentry) {
      Entry entry;
      const int index = rdfentry % batch_size;
      if(index >= start_idx && index < stop_idx)
        queue.Pop(entry);
      return entry;
    }, { "rdfentry_" });
    return df_out;
  }

  void join()
  {
    if(thread)
      thread->join();
  }

  EntryQueue<Entry> queue;
  std::unique_ptr<std::thread> thread;
};

} // namespace analysis