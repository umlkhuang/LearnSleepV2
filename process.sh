#!/bin/bash

for entry in data/*.zip 
do 
	echo "Unzipping file: $entry" 
	unzip $entry -d data 
	fname=`echo $entry | cut -f1 -d'.'`
	mv data/SleepCollector.db $fname.db 
	echo "done"
done

