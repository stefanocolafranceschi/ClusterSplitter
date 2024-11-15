// -*- C++ -*-
//
// Package:    RecoLocalTracker/trial
// Class:      trial
//
/**\class trial trial.cc RecoLocalTracker/trial/plugins/trial.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Stefano Colafranceschi
//         Created:  Mon, 11 Nov 2024 22:44:44 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

// system include files
#include "TFile.h"
#include "TString.h"

// Alpaka headers
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

using namespace cms::alpakatools;

// Define the Alpaka accelerator type
using Acc = alpaka::acc::AccGpuCudaRt<alpaka::dim::Dim1, uint32_t>;
using DevAcc = alpaka::dev::Dev<Acc>;
using Queue = alpaka::queue::QueueCudaRtSync;  // For GPU

// Define a simple Alpaka kernel as a struct
struct DummyKernel {
  template <typename Acc, typename T>
  ALPAKA_FN_ACC void operator()(Acc const& acc, T* result) const {
    // Simple computation for each thread
    auto idx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
    result[idx] = idx * 2;  // Example dummy computation
  }
};

//
// class declaration
//

class trial : public edm::stream::EDProducer<> {
public:
  explicit trial(const edm::ParameterSet&);
  ~trial() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
std::string configString_;  // Holds the configuration string passed from Python
  TFile* rootFile_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
trial::trial(const edm::ParameterSet& iConfig)
    : configString_(iConfig.getParameter<std::string>("configString")) {
  // Initialize ROOT file and write configuration string
  rootFile_ = new TFile("config_output.root", "RECREATE");
}


trial::~trial() {
  if (rootFile_) {
    rootFile_->Close();
    delete rootFile_;
  }
}
//
// member functions
//

void trial::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // Define a device for the accelerator
    DevAcc devAcc = alpaka::pltf::getDevByIdx<DevAcc>(0);  // Retrieve device
    Queue queue(devAcc);  // Set up the queue on this device

    // Define work divisions (e.g., grid and block sizes)
    const uint32_t elements = 1024;  // Number of elements to process
    auto workDiv = alpaka::workdiv::WorkDivMembers<1>(elements, 1, 1);

    // Allocate host and device memory
    auto hostBuf = alpaka::allocBuf<uint32_t, uint32_t>(alpaka::pltf::getDevByIdx<alpaka::pltf::PlatformCpu>(0), elements);
    auto devBuf = alpaka::allocBuf<uint32_t, uint32_t>(devAcc, elements);

    // Set up the queue for async execution
    alpaka::queue::QueueCudaRtSync queue(devAcc);

    // Launch the kernel
    DummyKernel kernel;
    alpaka::exec<Acc>(queue, workDiv, kernel, devBuf.data());

    // Copy the results back to host memory
    alpaka::mem::view::copy(queue, hostBuf, devBuf, elements);
    alpaka::wait(queue);

    // Here, you can store results in a ROOT file or use them as needed
}


// ------------ method called once each stream before processing any runs, lumis or events  ------------
void trial::beginStream(edm::StreamID) {
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void trial::endStream() {
  // please remove this method if not needed
}

// ------------ method called when starting to processes a run  ------------
/*
void
trial::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
trial::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
trial::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
trial::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

void trial::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("configString", "default")->setComment("Configuration string to store in ROOT file");
  descriptions.add("trial", desc);
}

//define this as a plug-in
}
DEFINE_FWK_MODULE(trial);
