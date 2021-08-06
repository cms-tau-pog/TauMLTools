source /cvmfs/grid.desy.de/etc/profile.d/grid-ui-env.sh
source /cvmfs/cms.cern.ch/cmsset_default.sh
export CMSSW_GIT_REFERENCE=/cvmfs/cms.cern.ch/cmssw.git.daily
source /cvmfs/cms.cern.ch/crab3/crab.sh

cmsenv
#source /cvmfs/cms.cern.ch/common/crab-setup.sh
voms-proxy-init -rfc -voms cms -valid 192:00
export X509_USER_PROXY=`voms-proxy-info -path`

export work_dir=/nfs/dust/cms/user/mykytaua/dataDeepTau/stau_masspoints_v3
export out_dir=/store/user/myshched/Signal2018ctau_TauTuple_masspoints_v3

#echo "Submission of stau official"
#crab_submit.py --workArea $work_dir --cfg ./Production/python/Production.py --site T2_DE_DESY --output $out_dir ./Production/crab/configs/2018/LLSTau.txt 

#echo "Submission of stau privat"
crab_submit.py --inputDBS phys03 --workArea $work_dir --cfg ./Production/python/Production.py --site T2_DE_DESY --output $out_dir ./Production/crab/configs/2018/LLSTau_priv.txt --splitting FileBased --unitsPerJob 100
