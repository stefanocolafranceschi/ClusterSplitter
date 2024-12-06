import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 50

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1000))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:/data/user/tetto/12834.0_TTbar_14TeV+2024/step3.root")
)

process.trial = cms.EDProducer("trial",  # Corrected path to the plugin
    configString=cms.string("This is my configuration string"),
    nHits=cms.uint32(100),   # Match uint32_t in C++
    offset=cms.int32(10)     # Match int32_t in C++
)

process.p = cms.Path(process.trial)

