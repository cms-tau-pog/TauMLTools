/*! Definition of a tuple with summary information about production.
 */

#pragma once

#include "TauMLTools/Core/interface/SmartTree.h"

#define SUMMARY_DATA() \
    /* Run statistics */ \
    VAR(UInt_t, exeTime) \
    VAR(ULong64_t, numberOfProcessedEvents) \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_tuple, ProdSummary, SummaryTuple, SUMMARY_DATA, "summary")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_tuple, SummaryTuple, SUMMARY_DATA)
#undef VAR
#undef SUMMARY_DATA
