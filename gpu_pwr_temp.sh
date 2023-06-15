#!/bin/bash

logfile="lhntest.log"

if [ -f $logfile ];then
   echo "The $logfile  is already exist, now delete it "
   rm $logfile
fi
sleep 3
nohup hx-smi dmon --select mu -f $logfile  -l 5 &

echo "Begin to account 50 seconds"
sleep 50 

ps -ef | grep "hx-smi dmon" | grep -v grep | head -n 1 | awk '{print$2}' | xargs kill -9

cat $logfile  | grep -v [a-zA-z] | awk 'BEGIN {sum=0};{sum+=$3} END {print "--------gpu avg is: " sum/NR}'
cat $logfile  | grep -v [a-zA-z] | awk 'BEGIN {sum=0};{sum+=$12} END {print "--------mem avg is: " sum/NR}'


#cat $logfile  | grep -v [a-zA-z] | awk 'BEGIN {sum=0};{sum+=$4} END {print "--------temp avg is: " sum/NR}'

