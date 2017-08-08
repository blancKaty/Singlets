#!/bin/bash

make

cd target
echo "_________________Compute homographies _____________________"
./homographie "../$1"
echo "_________________Compute Singularities_____________________"
./singularity "../$1"
echo "_________________Compute Saturation________________________"
./saturation_Median "../$1"
echo "_________________Compute Singularities Histograms _____________________"
./quantityOfSing "../$1"
echo "_________________Compute Singlets_________________________"
./computeChains "../$1"
echo "_________________Compute short summary of salient moments_________________"
./saliantShort "../$1"
echo "___________________Compute Summary_________________"
./shortenizer "../$1" "../outputDoc/listeFrameSailants.txt"
cd ../outputDoc
rm saturation.txt segmentation.txt tailleHistoHist5.txt sauvegardChainesHist5.txt histo3d*.txt homographies*.txt
