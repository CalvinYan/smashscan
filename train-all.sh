#!/bin/sh

for a in smash.melee.exp*.tiff;
do
	BASENAME=`basename $a .tiff`;
	tesseract $a $BASENAME box.train
done

unicharset_extractor smash.melee.exp*.box

shapeclustering -F font_properties -U unicharset smash.melee.exp0.tr
mftraining -F font_properties -U unicharset -O smash.unicharset smash.melee.exp*.tr
cntraining smash.melee.exp*.tr

mkdir -p tessdata
cp unicharset tessdata/smash.unicharset
cp pffmtable tessdata/smash.pffmtable
cp normproto tessdata/smash.normproto
cp inttemp tessdata/smash.inttemp
cp shapetable tessdata/smash.shapetable

cd tessdata
combine_tessdata smash.
cd ..
read -rsp 'Press any key to continue...\n' -n1 key
