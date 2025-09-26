/** \class reco::GenStatusFlags
 The original code can be found at https://github.com/cms-sw/cmssw/blob/master/DataFormats/HepMCCandidate/interface/GenStatusFlags.h
 * enum for generator status flags */

#pragma once

#include <bitset>

class GenStatusFlags{
public:
    GenStatusFlags(){}
    GenStatusFlags(uint16_t flags_) :
        flags(flags_) {}

public:
    enum StatusBits {
        kIsPrompt = 0,
        kIsDecayedLeptonHadron = 1,
        kIsTauDecayProduct = 2,
        kIsPromptTauDecayProduct = 3,
        kIsDirectTauDecayProduct = 4,
        kIsDirectPromptTauDecayProduct = 5,
        kIsDirectHadronDecayProduct = 6,
        kIsHardProcess = 7,
        kFromHardProcess = 8,
        kIsHardProcessTauDecayProduct = 9,
        kIsDirectHardProcessTauDecayProduct = 10,
        kFromHardProcessBeforeFSR = 11,
        kIsFirstCopy = 12,
        kIsLastCopy = 13,
        kIsLastCopyBeforeFSR = 14
    };

    /////////////////////////////////////////////////////////////////////////////
    //these are robust, generator-independent functions for categorizing
    //mainly final state particles, but also intermediate hadrons/taus

    //is particle prompt (not from hadron, muon, or tau decay)
    bool isPrompt() const { return flags[kIsPrompt]; }
    void setIsPrompt(bool b) { flags[kIsPrompt] = b; }

    //is particle a decayed hadron, muon, or tau (does not include resonance decays like W,Z,Higgs,top,etc)
    //This flag is equivalent to status 2 in the current HepMC standard
    //but older generators (pythia6, herwig6) predate this and use status 2 also for other intermediate
    //particles/states
    bool isDecayedLeptonHadron() const { return flags[kIsDecayedLeptonHadron]; }
    void setIsDecayedLeptonHadron(bool b) { flags[kIsDecayedLeptonHadron] = b; }

    //this particle is a direct or indirect tau decay product
    bool isTauDecayProduct() const { return flags[kIsTauDecayProduct]; }
    void setIsTauDecayProduct(bool b) { flags[kIsTauDecayProduct] = b; }

    //this particle is a direct or indirect decay product of a prompt tau
    bool isPromptTauDecayProduct() const { return flags[kIsPromptTauDecayProduct]; }
    void setIsPromptTauDecayProduct(bool b) { flags[kIsPromptTauDecayProduct] = b; }

    //this particle is a direct tau decay product
    bool isDirectTauDecayProduct() const { return flags[kIsDirectTauDecayProduct]; }
    void setIsDirectTauDecayProduct(bool b) { flags[kIsDirectTauDecayProduct] = b; }

    //this particle is a direct decay product from a prompt tau
    bool isDirectPromptTauDecayProduct() const { return flags[kIsDirectPromptTauDecayProduct]; }
    void setIsDirectPromptTauDecayProduct(bool b) { flags[kIsDirectPromptTauDecayProduct] = b; }

    //this particle is a direct decay product from a hadron
    bool isDirectHadronDecayProduct() const { return flags[kIsDirectHadronDecayProduct]; }
    void setIsDirectHadronDecayProduct(bool b) { flags[kIsDirectHadronDecayProduct] = b; }

    /////////////////////////////////////////////////////////////////////////////
    //these are generator history-dependent functions for tagging particles
    //associated with the hard process
    //Currently implemented for Pythia 6 and Pythia 8 status codes and history
    //and may not have 100% consistent meaning across all types of processes
    //Users are strongly encouraged to stick to the more robust flags above

    //this particle is part of the hard process
    bool isHardProcess() const { return flags[kIsHardProcess]; }
    void setIsHardProcess(bool b) { flags[kIsHardProcess] = b; }

    //this particle is the direct descendant of a hard process particle of the same pdg id
    bool fromHardProcess() const { return flags[kFromHardProcess]; }
    void setFromHardProcess(bool b) { flags[kFromHardProcess] = b; }

    //this particle is a direct or indirect decay product of a tau
    //from the hard process
    bool isHardProcessTauDecayProduct() const { return flags[kIsHardProcessTauDecayProduct]; }
    void setIsHardProcessTauDecayProduct(bool b) { flags[kIsHardProcessTauDecayProduct] = b; }

    //this particle is a direct decay product of a tau
    //from the hard process
    bool isDirectHardProcessTauDecayProduct() const { return flags[kIsDirectHardProcessTauDecayProduct]; }
    void setIsDirectHardProcessTauDecayProduct(bool b) { flags[kIsDirectHardProcessTauDecayProduct] = b; }

    //this particle is the direct descendant of a hard process particle of the same pdg id
    //For outgoing particles the kinematics are those before QCD or QED FSR
    //This corresponds roughly to status code 3 in pythia 6
    bool fromHardProcessBeforeFSR() const { return flags[kFromHardProcessBeforeFSR]; }
    void setFromHardProcessBeforeFSR(bool b) { flags[kFromHardProcessBeforeFSR] = b; }

    //this particle is the first copy of the particle in the chain with the same pdg id
    bool isFirstCopy() const { return flags[kIsFirstCopy]; }
    void setIsFirstCopy(bool b) { flags[kIsFirstCopy] = b; }

    //this particle is the last copy of the particle in the chain with the same pdg id
    //(and therefore is more likely, but not guaranteed, to carry the final physical momentum)
    bool isLastCopy() const { return flags[kIsLastCopy]; }
    void setIsLastCopy(bool b) { flags[kIsLastCopy] = b; }

    //this particle is the last copy of the particle in the chain with the same pdg id
    //before QED or QCD FSR
    //(and therefore is more likely, but not guaranteed, to carry the momentum after ISR)
    bool isLastCopyBeforeFSR() const { return flags[kIsLastCopyBeforeFSR]; }
    void setIsLastCopyBeforeFSR(bool b) { flags[kIsLastCopyBeforeFSR] = b; }

    const std::bitset<15>& getFlags() const {return flags;}
 private:
     std::bitset<15> flags;
};