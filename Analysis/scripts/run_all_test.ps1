$in_dir = ".\tuples\remote_testing"
$out_dir = ".\results\0610_s3"
$model_file = ".\20L1024N_0610_s3_loss.hdf5.pb"

#$samples = @("DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8", `
#             "TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8", `
#             "TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8", `
#             "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8")

#$samples = @( `
#             "TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8", `
#             "TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8", `
#             "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8")

$samples = @( "ZeroBias")

foreach( $sample in $samples) {
    Write-Output "Processing $sample ..."
    python .\Analysis\python\apply_training.py --input $in_dir\$sample `
           --output $out_dir\$sample.hdf5 --model $model_file
}
