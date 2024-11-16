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

// Alpaka headers
#include <alpaka/alpaka.hpp>
#include <alpaka/acc/AccGpuCudaRt.hpp>  // Add this for CUDA support

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

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
#include <memory>
#include <iostream>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
    using Acc = acc::AccGpuCudaRt<dim::Dim1, uint32_t>;  // Dim1 refers to a 1D grid, uint32_t for indices
    using DevAcc = dev::Dev<Acc>;  // Define the device based on the accelerator type
    using Queue = queue::QueueCudaRtSync;  // For GPU queues

    // Define a simple Alpaka kernel as a struct
    struct DummyKernel {
        template <typename Acc, typename T>
        ALPAKA_FN_ACC void operator()(Acc const& acc, T* result) const {
            // Simple computation for each thread
            auto idx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            result[idx] = idx * 2;  // Example dummy computation
        }
    };

    class trial : public edm::stream::EDProducer<> {
    public:
        explicit trial(const edm::ParameterSet&);
        ~trial() override;

        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
        void beginStream(edm::StreamID) override;
        void produce(edm::Event&, const edm::EventSetup&) override;
        void endStream() override;

        // member data
        std::string configString_;  // Holds the configuration string passed from Python
        TFile* rootFile_;
    };

    // Constructor
    trial::trial(const edm::ParameterSet& iConfig)
        : configString_(iConfig.getParameter<std::string>("configString")) {
        // Initialize ROOT file and write configuration string
        rootFile_ = new TFile("config_output.root", "RECREATE");
    }

    // Destructor
    trial::~trial() {
        if (rootFile_) {
            rootFile_->Close();
            delete rootFile_;
        }
    }

    // produce method (updated version)
    void trial::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
        // Define a device for the accelerator
        DevAcc devAcc = alpaka::pltf::getDevByIdx<DevAcc>(0);  // Retrieve device
        alpaka::queue::QueueCudaRtSync queue(devAcc);  // Set up the queue on this device

        // Define work divisions (e.g., grid and block sizes)
        const uint32_t elements = 1024;  // Number of elements to process
        auto workDiv = alpaka::workdiv::WorkDivMembers<1>(elements, 1, 1);

        // Use simple uint32_t buffer for memory management
        auto hostBuf = alpaka::allocBuf<uint32_t>(alpaka::pltf::getDevByIdx<alpaka::pltf::PlatformCpu>(0), elements);
        auto devBuf = alpaka::allocBuf<uint32_t>(devAcc, elements);

        // Launch the kernel (DummyKernel is assumed to be defined earlier)
        DummyKernel kernel;
        alpaka::exec<Acc>(queue, workDiv, kernel, devBuf.data());

        // Copy the results back to host memory
        alpaka::mem::view::copy(queue, hostBuf, devBuf, elements);
        alpaka::wait(queue);

        // For debugging, output the result
        for (size_t i = 0; i < elements; ++i) {
            std::cout << "Result[" << i << "] = " << hostBuf[i] << std::endl;
        }
    }

    // Fill descriptions for the configuration
    void trial::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<std::string>("configString", "default")->setComment("Configuration string to store in ROOT file");
        descriptions.add("trial", desc);
    }
}
// Define this as a plug-in
DEFINE_FWK_MODULE(ALPAKA_ACCELERATOR_NAMESPACE::trial);
