#!/bin/bash

# Assign the filename
templatename='config_template.json'
sRUN='RUN'
name='config_'
ext='.json'
n1='_theta'
n2='_sgd'
n3='_lam'
n4='_run'
n5='_damp'
n6='_batch'
n7='_epochs'
n8='_directStep'
n9='_updateEvery'

#Template variables
sBETA='BETA'
sALG='ALG'
sCOMMIT_NR='COMMIT_NR'
sSIGMA_FIXED='SIGMA_FIXED'
sDIRECT_STEP='DIRECT_STEP'
sNU_THETA='NU_THETA'
sNU_SGD='NU_SGD'
sNU_LAMBDA='NU_LAMBDA'
sRUN='RUN'
sDAMPING='DAMPING'
sBATCH_SIZE='BATCH_SIZE'
sBATCH_SIZE_AUX='bATCH_SIZE_AUX'
sTARGET='TARGET'
sSIGMA_INIT='SIGMA_INIT'
sEPOCHS='EPOCHS'
sSEED_INIT='SEED_INIT'
sUPDATE_EVERY='UPDATE_EVERY'

#Fixed values
rBETA='0.9'
rALG='"FL"'
rDIRECT_STEP='"True"'
rCOMMIT_NR='"961fef0"'
rBATCH_SIZE='1000'
rBATCH_SIZE_AUX='1000'
rTARGET='"FACES"'
rSIGMA_FIXED='"True"'
rSIGMA_INIT='1.0'
rEPOCHS='600'
rSEED_INIT='12'

#List of values to loop
rRUNs='1 2 3 4 5 6 7 8 9 10'
rNU_THETAs='0.05'
rNU_SGDs='0.01'
rNU_LAMBDAs='1e-4 1e-3 1e-2'
rDAMPINGs='0.001 0.0001'
rUPDATE_EVERYs='10'

for rUPDATE_EVERY in $rUPDATE_EVERYs; do
 for rRUN in $rRUNs; do
  for rNU_THETA in $rNU_THETAs; do
   for rNU_SGD in $rNU_SGDs; do
    for rNU_LAMBDA in $rNU_LAMBDAs; do
     for rDAMPING in $rDAMPINGs; do
         filename="$name${rTARGET:1:-1}$n1$rNU_THETA$n2$rNU_SGD$n3$rNU_LAMBDA$n4$rRUN$n5$rDAMPING$n6$rBATCH_SIZE$n7$rEPOCHS$n8${rDIRECT_STEP:1:-1}$n9$rUPDATE_EVERY$ext"
         cp $templatename $filename
         sed -i "s/$sNU_THETA/$rNU_THETA/" $filename
         sed -i "s/$sNU_SGD/$rNU_SGD/" $filename
         sed -i "s/$sNU_LAMBDA/$rNU_LAMBDA/" $filename
         sed -i "s/$sRUN/$rRUN/" $filename
         sed -i "s/$sDAMPING/$rDAMPING/" $filename
         sed -i "s/$sBETA/$rBETA/" $filename
         sed -i "s/$sALG/$rALG/" $filename
         sed -i "s/$sCOMMIT_NR/$rCOMMIT_NR/" $filename
         sed -i "s/$sSIGMA_FIXED/$rSIGMA_FIXED/" $filename
         sed -i "s/$sDIRECT_STEP/$rDIRECT_STEP/" $filename
         sed -i "s/$sBATCH_SIZE/$rBATCH_SIZE/" $filename
         sed -i "s/$sBATCH_SIZE_AUX/$rBATCH_SIZE_AUX/" $filename
         sed -i "s/$sTARGET/$rTARGET/" $filename
         sed -i "s/$sSIGMA_INIT/$rSIGMA_INIT/" $filename
         sed -i "s/$sUPDATE_EVERY/$rUPDATE_EVERY/" $filename
         sed -i "s/$sEPOCHS/$rEPOCHS/" $filename
         sed -i "s/$sSEED_INIT/$rSEED_INIT/" $filename
     done
    done
   done
  done
 done
done